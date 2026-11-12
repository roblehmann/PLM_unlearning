"""Build full-length (fullchain) UniProt protein datasets to complement the
domain-clipped F_train/F_test.fasta used by experiments/0_F_candidate_PPL and
experiments/1_naive_PPL_Floor.

For each pairN/F_{train,test}.fasta (Pfam domain-envelope sequences, e.g.
">ACC_SPECIES/12-345 ACC.1 PFxxxxx;Name;"), this:

  1. extracts the unique UniProt accession per sequence (2nd header token,
     version stripped), subsamples to a bounded pool (default 12000 train /
     1500 test -- comfortably above floor_ppl.py's own caps of 8192/256 so
     downstream subsampling still has room), and dedupes;
  2. fetches the full-length UniProtKB sequence for each accession via the
     REST batch endpoint (100 accessions/request, retried with backoff);
  3. decontaminates: mmseqs easy-search F_test_full vs F_train_full and drops
     any test sequence >= --max-identity to a train sequence, mirroring the
     t1=0.5 leakage floor used elsewhere in this project (see
     src/data/leakage_filter.py).

Writes experiments/0_F_candidate_PPL/data_fullchain/pairN/{F_train,F_test}.fasta
plus manifest.json (accession resolution, decontam counts). Domain data under
data/pairN is untouched; naive_ppl.py / floor_ppl.py need no code changes --
point them at data_fullchain via --data-dir (and a fresh --naive-csv from a
naive_ppl.py run over the same --data-dir).

    python experiments/0_F_candidate_PPL/build_fullchain.py
    python experiments/0_F_candidate_PPL/build_fullchain.py --pairs pair1 pair2
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, read_fasta, save_json, write_fasta  # noqa: E402

UNIPROT_BATCH_URL = "https://rest.uniprot.org/uniprotkb/accessions"
BATCH_SIZE = 100
MAX_RETRIES = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(HERE / "data"), help="dir containing pairN/F_{train,test}.fasta")
    p.add_argument("--out-dir", default=str(HERE / "data_fullchain"))
    p.add_argument("--pairs", nargs="*", default=None, help="restrict to these pair dir names (default: all)")
    p.add_argument("--max-train", type=int, default=12000, help="accession pool cap before fetch")
    p.add_argument("--max-test", type=int, default=1500, help="accession pool cap before fetch")
    p.add_argument("--max-identity", type=float, default=0.50, help="t1 leakage ceiling for test-vs-train decontam")
    p.add_argument("--coverage", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--pause", type=float, default=0.34, help="seconds between UniProt batch requests")
    return p.parse_args()


def domain_accessions(fasta_path: Path) -> list[str]:
    """Unique UniProt accessions in header-appearance order (2nd token, version stripped)."""
    seen, accs = set(), []
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            parts = line[1:].split()
            if len(parts) < 2:
                continue
            acc = parts[1].split(".")[0]
            if acc not in seen:
                seen.add(acc)
                accs.append(acc)
    return accs


def iter_fasta_text(text: str):
    header, chunks = None, []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(chunks)
            header, chunks = line[1:], []
        else:
            chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def fetch_fullchain(accessions: list[str], batch_size: int, pause: float) -> dict[str, tuple[str, str]]:
    """acc -> (fasta_header, seq). Missing accessions (obsolete/merged) are silently absent."""
    out: dict[str, tuple[str, str]] = {}
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i : i + batch_size]
        params = {"accessions": ",".join(batch), "format": "fasta"}
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(UNIPROT_BATCH_URL, params=params, timeout=30)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)
        for header, seq in iter_fasta_text(r.text):
            # header e.g. "sp|P69905|HBA_HUMAN ..." or "tr|A0A395MHW1|A0A395MHW1_9HYPO ..."
            acc = header.split("|")[1] if "|" in header else header.split()[0]
            out[acc] = (header, seq)
        print(f"    fetched {min(i + batch_size, len(accessions))}/{len(accessions)} "
              f"({len(out)} resolved so far)", end="\r")
        time.sleep(pause)
    print()
    return out


def subsample(seq: list[str], k: int, seed: int) -> list[str]:
    if len(seq) <= k:
        return seq
    rng = random.Random(seed)
    return rng.sample(seq, k)


def mmseqs_decontam(test_fasta: Path, train_fasta: Path, max_identity: float, coverage: float, threads: int) -> tuple[set[str], float]:
    """Return (accessions to drop from test, max observed identity)."""
    if shutil.which("mmseqs") is None:
        raise SystemExit("mmseqs not on PATH -- `module load mmseqs2/14.7` first")
    tmp_root = Path(tempfile.mkdtemp(prefix="fullchain_decontam_"))
    res = tmp_root / "result.m8"
    tmp = tmp_root / "tmp"
    cmd = [
        "mmseqs", "easy-search", str(test_fasta), str(train_fasta), str(res), str(tmp),
        "--min-seq-id", "0.0", "-c", str(coverage), "--cov-mode", "0",
        "--format-output", "query,target,fident", "--threads", str(threads), "-v", "1",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    to_drop, max_id = set(), 0.0
    with open(res) as fh:
        for line in fh:
            q, _t, fident = line.rstrip("\n").split("\t")[:3]
            fident = float(fident)
            max_id = max(max_id, fident)
            if fident >= max_identity:
                to_drop.add(q)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return to_drop, max_id


def build_pair(pair_dir: Path, out_dir: Path, args) -> dict:
    pair = pair_dir.name
    print(f"[{pair}] parsing domain accessions ...")
    train_accs_all = domain_accessions(pair_dir / "F_train.fasta")
    test_accs_all = domain_accessions(pair_dir / "F_test.fasta")
    train_accs = subsample(train_accs_all, args.max_train, args.seed)
    test_accs = subsample(test_accs_all, args.max_test, args.seed)
    print(f"    train pool {len(train_accs)}/{len(train_accs_all)}, test pool {len(test_accs)}/{len(test_accs_all)}")

    print(f"[{pair}] fetching full-length train sequences from UniProt ...")
    train_resolved = fetch_fullchain(train_accs, args.batch_size, args.pause)
    print(f"[{pair}] fetching full-length test sequences from UniProt ...")
    test_resolved = fetch_fullchain(test_accs, args.batch_size, args.pause)

    pair_out = ensure_dir(out_dir / pair)
    train_raw = pair_out / "F_train_full.raw.fasta"
    test_raw = pair_out / "F_test_full.raw.fasta"
    write_fasta(((acc, seq) for acc, (_h, seq) in train_resolved.items()), train_raw)
    write_fasta(((acc, seq) for acc, (_h, seq) in test_resolved.items()), test_raw)

    print(f"[{pair}] decontaminating (mmseqs, t1={args.max_identity}) ...")
    to_drop, max_id = mmseqs_decontam(test_raw, train_raw, args.max_identity, args.coverage, args.threads)
    kept = [(acc, seq) for acc, (_h, seq) in test_resolved.items() if acc not in to_drop]

    write_fasta(((acc, seq) for acc, (_h, seq) in train_resolved.items()), pair_out / "F_train.fasta")
    write_fasta(kept, pair_out / "F_test.fasta")
    train_raw.unlink()
    test_raw.unlink()

    manifest = {
        "pair": pair,
        "train_pool": len(train_accs), "train_pool_available": len(train_accs_all),
        "train_resolved": len(train_resolved), "train_unresolved": len(train_accs) - len(train_resolved),
        "test_pool": len(test_accs), "test_pool_available": len(test_accs_all),
        "test_resolved": len(test_resolved), "test_unresolved": len(test_accs) - len(test_resolved),
        "test_dropped_decontam": len(to_drop), "test_kept": len(kept),
        "max_identity_observed": max_id, "max_identity_threshold": args.max_identity,
        "coverage": args.coverage,
    }
    save_json(manifest, pair_out / "manifest.json")
    print(f"[{pair}] train={len(train_resolved)} test={len(kept)} "
          f"(dropped {len(to_drop)}/{len(test_resolved)} decontam, max_id_observed={max_id:.3f})")
    return manifest


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    pair_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and (d / "F_train.fasta").exists())
    if args.pairs:
        pair_dirs = [d for d in pair_dirs if d.name in set(args.pairs)]
    if not pair_dirs:
        raise SystemExit(f"no pair dirs with F_train.fasta under {data_dir}")

    manifests = [build_pair(d, out_dir, args) for d in pair_dirs]
    save_json(manifests, out_dir / "build_manifest.json")
    print(f"\nWrote fullchain datasets for {len(manifests)} pair(s) under {out_dir}")


if __name__ == "__main__":
    main()
