"""Homology-controlled STATIC re-score across every run root (no training, no attacks).

Motivation. The project has three different datasets all called "PF00106", built by two
different pipelines, with F_test sizes 6000 / 2053 / 1554. Only two of them are homology-
controlled. Published block factors are therefore not comparable across roots, because a
held-out set that shares >=50% identity with F_train measures memorisation of the training
cloud as much as family capability.

This script puts every root on ONE criterion and re-scores the STATIC block only:

  1. decontaminate  F_test against F_train with mmseqs2 easy-search
     (--min-seq-id 0.5 -c 0.5 --cov-mode 0 -s 7.5); keep F_test sequences with NO hit.
     Both sides are re-written under synthetic ids q0,q1,... because mmseqs derives its own
     id from the FASTA header and UniProt `tr|ACC|NAME` headers silently break a
     string join (the bug that invalidated the first experiment-4 run).
  2. re-score theta_0 (base A), A' and every available B on the native and the
     decontaminated held-out set, with the project's own eval settings, and report
     block = PPL_model / PPL_base on each.

What this CANNOT do: re-score the attack/durability curves. The sweep never persists per-cell
checkpoints (src/sweep/run_sweep.py passes no save_model_path), so every multiplier M and
recovery curve R(c) in the manuscript stays on its native split and CANNOT be moved to the
decontaminated set without re-running the sweep. Only the static factors below are
decontaminated. Say so wherever the two are reported together.

CPU (mmseqs) + one GPU, ~1h for all roots.

    python scripts/11_decon_static_rescore.py                 # all roots
    python scripts/11_decon_static_rescore.py --roots mgnify  # one root
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_ID = "facebook/esm2_t12_35M_UR50D"
N_EVAL, EVAL_BATCH, MASKS_PER_PASS, SEED = 256, 64, 8, 0

# (key, root dir relative to repo root, label, [(model label, checkpoint dir), ...])
ROOTS = [
    ("pfam_legacy", ".", "PF00106 legacy — random 70/30 split, full-length", [
        ("A_prime", "checkpoints/A_prime"),
        ("B_mlac_fo", "checkpoints/B"),
        ("B_cir_peak", "checkpoints/cir35m_peak/B"),
        ("B_cir_lowdis", "checkpoints/cir35m_lowdis/B"),
        ("theta_acq", "checkpoints/acquired_F"),
    ]),
    ("pf00106_domain", "runs/PF00106_domain", "PF00106 domain-clipped — homology-aware split", [
        ("A_prime", "checkpoints/A_prime"),
        ("B_mlac_fo", "checkpoints/B"),
    ]),
    ("pf00106_fullchain", "runs/PF00106_fullchain", "PF00106 full-length — homology-aware split", [
        ("A_prime", "checkpoints/A_prime"),
        ("B_mlac_fo", "checkpoints/B"),
        ("B_cir_peak", "cir_peak/checkpoints/B"),
        ("B_cir_lowdis", "cir_lowdis/checkpoints/B"),
    ]),
    ("pf00107_fullchain", "runs/PF00107_fullchain", "PF00107 full-length — homology-aware split", [
        ("A_prime", "checkpoints/A_prime"),
        ("B_mlac_fo", "checkpoints/B"),
    ]),
    ("mgnify", "runs/MGYF18690_metagenomic", "MGYF0000018690 metagenomic — source-aware split", [
        ("A_prime", "checkpoints/A_prime"),
        ("B_mlac_fo", "checkpoints/B"),
        ("B_cir_best", "checkpoints/cir_knob/retain_pca_n24_b3-8/B"),
        ("theta_acq", "checkpoints/acquired_F"),
    ]),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="*", default=None, help="subset of root keys (default: all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-seq-id", type=float, default=0.5)
    p.add_argument("--coverage", type=float, default=0.5)
    p.add_argument("--sensitivity", type=float, default=7.5)
    p.add_argument("--n-eval", type=int, default=N_EVAL)
    p.add_argument("--force-decon", action="store_true", help="recompute cached decon FASTAs")
    p.add_argument("--out", default="docs/decon_reconciliation.json")
    return p.parse_args()


def _write_synthetic(records, path: Path) -> None:
    """Write records under synthetic ids q0,q1,... (list index) so mmseqs' own id
    derivation cannot desynchronise the join. See module docstring."""
    with open(path, "w") as fh:
        for i, r in enumerate(records):
            fh.write(f">q{i}\n{r.seq}\n")


def contaminated_indices(query_records, target_records, args, workdir: Path) -> set:
    """Indices of query records with >=1 target hit at >= min_seq_id and >= coverage."""
    q, t, out, tmp = (workdir / "q.fasta", workdir / "t.fasta", workdir / "hits.m8", workdir / "tmp")
    _write_synthetic(query_records, q)
    _write_synthetic(target_records, t)
    tmp.mkdir(exist_ok=True)
    cmd = ["mmseqs", "easy-search", str(q), str(t), str(out), str(tmp),
           "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage), "--cov-mode", "0",
           "-s", str(args.sensitivity), "--threads", str(os.cpu_count() or 8), "-v", "1"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    hits = set()
    with open(out) as fh:
        for line in fh:
            qid = line.split("\t", 1)[0].strip()
            if qid.startswith("q"):
                hits.add(int(qid[1:]))
    return hits


def decontaminate(root: Path, args) -> dict:
    """Cache <root>/data/splits/F_test_decon_c05.fasta + a manifest; return the manifest."""
    from src.utils.io import read_fasta, write_fasta

    splits = root / "data" / "splits"
    f_test_p, f_train_p = splits / "F_test.fasta", splits / "F_train.fasta"
    out_fa = splits / "F_test_decon_c05.fasta"
    out_js = splits / "F_test_decon_c05.json"
    if out_fa.exists() and out_js.exists() and not args.force_decon:
        return json.load(open(out_js))
    f_test, f_train = read_fasta(f_test_p), read_fasta(f_train_p)
    with tempfile.TemporaryDirectory(dir=str(splits)) as wd:
        hit = contaminated_indices(f_test, f_train, args, Path(wd))
    kept = [r for i, r in enumerate(f_test) if i not in hit]
    assert len(kept) + len(hit) == len(f_test), "kept + contaminated != input"
    write_fasta(kept, out_fa)
    man = {"n_F_test": len(f_test), "n_F_train": len(f_train), "n_contaminated": len(hit),
           "n_kept": len(kept), "frac_contaminated": len(hit) / max(1, len(f_test)),
           "min_seq_id": args.min_seq_id, "coverage": args.coverage,
           "sensitivity": args.sensitivity, "path": str(out_fa)}
    json.dump(man, open(out_js, "w"), indent=2)
    return man


def main() -> int:
    args = parse_args()
    if shutil.which("mmseqs") is None:
        raise SystemExit("mmseqs not on PATH (module load mmseqs2/14.7)")
    from src.data.datamodule import subsample
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.models.load import load_checkpoint, load_tokenizer
    from src.utils.io import fasta_seqs

    import torch

    tok = load_tokenizer(MODEL_ID)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    selected = [r for r in ROOTS if args.roots is None or r[0] in args.roots]
    out_all = {"criterion": {"min_seq_id": args.min_seq_id, "coverage": args.coverage,
                             "cov_mode": 0, "sensitivity": args.sensitivity,
                             "tool": "mmseqs2 easy-search", "rule": "drop F_test seq with any F_train hit"},
               "eval": {"n_eval": args.n_eval, "batch": EVAL_BATCH,
                        "masks_per_pass": MASKS_PER_PASS, "seed": SEED},
               "scope": "STATIC block factors only — attack/durability curves cannot be "
                        "re-scored (no per-cell sweep checkpoints exist)",
               "roots": {}}

    for key, rel, label, models in selected:
        root = ROOT / rel
        print(f"\n=== {key}  ({label})")
        man = decontaminate(root, args)
        print(f"  decon: {man['n_kept']}/{man['n_F_test']} kept "
              f"({100*man['frac_contaminated']:.1f}% train-homologous)")
        native_all = fasta_seqs(root / "data" / "splits" / "F_test.fasta")
        decon_all = fasta_seqs(Path(man["path"]))
        native = subsample(native_all, args.n_eval, seed=SEED)
        decon = subsample(decon_all, args.n_eval, seed=SEED) if decon_all else []

        def score(path):
            m = load_checkpoint(path, device=device).eval()
            pn = pseudo_perplexity(m, tok, native, device, EVAL_BATCH, 1024,
                                   masks_per_pass=MASKS_PER_PASS).ppl
            pdn = (pseudo_perplexity(m, tok, decon, device, EVAL_BATCH, 1024,
                                     masks_per_pass=MASKS_PER_PASS).ppl if decon else float("nan"))
            del m
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            return pn, pdn

        rows = {}
        bn, bd = score(MODEL_ID)
        rows["A_base"] = {"ppl_native": bn, "ppl_decon": bd, "checkpoint": MODEL_ID}
        print(f"  {'A_base':<14} native={bn:9.2f}  decon={bd:9.2f}")
        for mlabel, ckpt in models:
            p = root / ckpt
            if not (p / "config.json").exists():
                print(f"  {mlabel:<14} absent ({ckpt}) — skipped")
                continue
            pn, pdn = score(str(p))
            rows[mlabel] = {"ppl_native": pn, "ppl_decon": pdn, "checkpoint": str(Path(rel) / ckpt),
                            "block_native": pn / bn, "block_decon": pdn / bd if bd == bd else float("nan")}
            print(f"  {mlabel:<14} native={pn:9.2f}  decon={pdn:9.2f}   "
                  f"block {pn/bn:9.1f}x -> {pdn/bd:8.1f}x   shrink {(pn/bn)/(pdn/bd):6.1f}x")
        out_all["roots"][key] = {"label": label, "root": rel, "decontamination": man,
                                 "n_scored_native": len(native), "n_scored_decon": len(decon),
                                 "models": rows}
        json.dump(out_all["roots"][key], open(root / "results" / "decon_static.json", "w"), indent=2)

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_all, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
