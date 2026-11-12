"""Acquire the raw sequence data: family F, sibling F' (same Pfam clan), retain set
R (UniRef50 subsample), and a background set for the membership probe.

Two modes:
  * real  — query InterPro/Pfam + UniProt REST and download. Endpoints are stable
            as of writing but should be *confirmed* in Phase 1; they
            are isolated in the constants below.
  * toy   — synthesize tiny, motif-structured sequences entirely offline. Used by
            the smoke test and unit tests; pass data.toy=true.

Nothing here touches hazardous sequences: F and F' are benign Pfam families chosen
per the selection criteria in configs/data/pfam_default.yaml.
"""
from __future__ import annotations

import gzip
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.io import SeqRecord, ensure_dir, write_fasta

# --- External endpoints (CONFIRM in Phase 1) ------------------------------- #
INTERPRO_API = "https://www.ebi.ac.uk/interpro/api"
INTERPRO_WWWAPI = "https://www.ebi.ac.uk/interpro/wwwapi"
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIREF_STREAM = "https://rest.uniprot.org/uniref/stream"
AA = "ACDEFGHIKLMNPQRSTVWY"


# --------------------------------------------------------------------------- #
# HTTP helper
# --------------------------------------------------------------------------- #
def _http_get(url: str, params: Optional[dict] = None, stream: bool = False, retries: int = 3, timeout: int = 60):
    import requests

    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, stream=stream, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last!r})")


def _read_fasta_stream(resp, max_n: Optional[int]) -> List[SeqRecord]:
    """Parse a (possibly very large) FASTA HTTP stream, stopping after max_n records."""
    records: List[SeqRecord] = []
    header, chunks = None, []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(SeqRecord(header, "".join(chunks)))
                if max_n is not None and len(records) >= max_n:
                    return records
            header = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if header is not None and (max_n is None or len(records) < max_n):
        records.append(SeqRecord(header, "".join(chunks)))
    return records


# --------------------------------------------------------------------------- #
# Real downloads (InterPro / UniProt)
# --------------------------------------------------------------------------- #
def clan_families(clan_id: str) -> List[str]:
    """Return the Pfam family accessions belonging to a clan."""
    url = f"{INTERPRO_API}/set/pfam/{clan_id}/entry/pfam/"
    out: List[str] = []
    next_url: Optional[str] = url
    while next_url:
        data = _http_get(next_url).json()
        for item in data.get("results", []):
            acc = item.get("metadata", {}).get("accession") or item.get("accession")
            if acc:
                out.append(acc.upper())
        next_url = data.get("next")
    return out


def download_family_hmm(pfam_id: str, out_path: str) -> str:
    """Download the family profile HMM (used for homolog generation + scoring)."""
    url = f"{INTERPRO_WWWAPI}/entry/pfam/{pfam_id}/?annotation=hmm"
    resp = _http_get(url, stream=True)
    raw = resp.content
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass  # already plain text
    out_path = str(out_path)
    ensure_dir(Path(out_path).parent)
    with open(out_path, "wb") as fh:
        fh.write(raw)
    return out_path


def download_family_sequences(pfam_id: str, out_fasta: str, max_n: Optional[int] = None) -> int:
    """Download UniProtKB sequences annotated with the given Pfam entry."""
    resp = _http_get(UNIPROT_STREAM, params={"query": f"xref:pfam-{pfam_id}", "format": "fasta"}, stream=True)
    records = _read_fasta_stream(resp, max_n)
    return write_fasta(records, out_fasta)


def download_uniref50_subsample(n: int, out_fasta: str, query: str = "identity:0.5", seed: int = 0) -> int:
    """Stream a UniRef50 subsample. NOTE: this takes the first ``n`` streamed records,
    which is *not* a uniform sample. For production, shuffle a local UniRef50 release
    instead."""
    resp = _http_get(UNIREF_STREAM, params={"query": query, "format": "fasta"}, stream=True)
    records = _read_fasta_stream(resp, n)
    return write_fasta(records, out_fasta)


# --------------------------------------------------------------------------- #
# Toy / offline synthesis
# --------------------------------------------------------------------------- #
def _random_seq(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(AA) for _ in range(length))


def _motif_family(rng: random.Random, n: int, motif: str, length: int, conserve: float) -> List[SeqRecord]:
    """A toy 'family': random sequences carrying a shared motif at a fixed offset,
    with ``conserve`` fraction of motif columns preserved per sequence."""
    recs = []
    offset = length // 3
    for i in range(n):
        s = list(_random_seq(rng, length))
        for j, aa in enumerate(motif):
            if rng.random() < conserve:
                s[offset + j] = aa
        recs.append(SeqRecord(f"toy_{motif[:3]}_{i}", "".join(s)))
    return recs


def synthesize_toy(out_dir: str, cfg, seed: int = 0) -> Dict[str, str]:
    """Create tiny offline F, F', R, background + placeholder HMMs. Returns paths.

    F and F' share part of a motif (siblings in a 'clan'); R and background are
    generic. A handful of near-duplicates of F are injected into R so the leakage
    filter has something to remove.
    """
    rng = random.Random(seed)
    out = ensure_dir(out_dir)
    raw = ensure_dir(out / "raw")
    L = 60

    f_motif = "WACDEFGHIK"
    fp_motif = "WACDEFmnpq".upper()  # shares the 'WACDEF' prefix with F
    n_fam = 120
    F = _motif_family(rng, n_fam, f_motif, L, conserve=0.95)
    Fp = _motif_family(rng, n_fam, fp_motif, L, conserve=0.95)

    retain_cfg = getattr(cfg, "retain", {})
    requested = retain_cfg.get("n_sequences", 400) if hasattr(retain_cfg, "get") else getattr(retain_cfg, "n_sequences", 400)
    n_retain = min(max(64, int(requested)), 600)  # keep the toy set tiny regardless of the configured size
    R = [SeqRecord(f"retain_{i}", _random_seq(rng, L)) for i in range(n_retain)]
    # inject near-duplicates of F into R (these MUST be removed by the leakage filter)
    contaminants = []
    for i in range(8):
        src = list(F[i].seq)
        for _ in range(2):  # a couple of point mutations -> still highly similar
            src[rng.randrange(L)] = rng.choice(AA)
        contaminants.append(SeqRecord(f"retain_contam_F_{i}", "".join(src)))
    R += contaminants

    n_bg = 80
    background = [SeqRecord(f"bg_{i}", _random_seq(rng, L)) for i in range(n_bg)]

    paths = {
        "F_seqs": str(raw / "F" / "family.fasta"),
        "Fp_seqs": str(raw / "F_prime" / "family.fasta"),
        "F_hmm": str(raw / "F" / "family.hmm"),
        "Fp_hmm": str(raw / "F_prime" / "family.hmm"),
        "retain": str(raw / "retain" / "uniref50_subsample.fasta"),
        "background": str(raw / "background" / "background.fasta"),
        "contaminant_ids": [c.id for c in contaminants],
    }
    write_fasta(F, paths["F_seqs"])
    write_fasta(Fp, paths["Fp_seqs"])
    write_fasta(R, paths["retain"])
    write_fasta(background, paths["background"])
    # placeholder "HMM" files: store the motif + seeds so the toy homolog generator can use them
    for hmm_path, motif, fam in ((paths["F_hmm"], f_motif, F), (paths["Fp_hmm"], fp_motif, Fp)):
        ensure_dir(Path(hmm_path).parent)
        with open(hmm_path, "w") as fh:
            fh.write(f"# TOY-HMM motif={motif} length={L}\n")
    return paths
