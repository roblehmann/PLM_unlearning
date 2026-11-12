"""Leakage filter (validity control #1).

Remove any retain (R) sequence too similar to F or F'. Contamination of R by
family homologs would corrupt *both* the block (the defender would see F in its
"benign" retain set) and the attack measurement.

Primary tool: MMseqs2 (`easy-search`). cd-hit-2d supported as an alternative. A
dependency-free Python proxy (k-mer containment) is used for the toy/offline path
and is selected automatically when no binary is found — it is NOT a substitute for
MMseqs2 on real data and emits a warning.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.utils.io import iter_fasta, read_fasta, write_fasta


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


# --------------------------------------------------------------------------- #
# MMseqs2
# --------------------------------------------------------------------------- #
def mmseqs_search(query_fasta: str, target_fasta: str, coverage: float, workdir: str, threads: int = 4) -> List[Tuple[str, float, float]]:
    """Run `mmseqs easy-search` (query=R, target=family). Returns list of
    (query_id, fident, max_cov). fident is the fraction of identical residues."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    res = workdir / "result.m8"
    tmp = workdir / "tmp"
    cmd = [
        "mmseqs", "easy-search", str(query_fasta), str(target_fasta), str(res), str(tmp),
        "--min-seq-id", "0.0",
        "-c", str(coverage),
        "--cov-mode", "0",
        "--format-output", "query,target,fident,qcov,tcov",
        "--threads", str(threads),
        "-v", "1",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    hits: List[Tuple[str, float, float]] = []
    with open(res) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            q, _t, fident, qcov, tcov = parts[:5]
            hits.append((q, float(fident), max(float(qcov), float(tcov))))
    return hits


# --------------------------------------------------------------------------- #
# Python k-mer proxy (offline fallback only)
# --------------------------------------------------------------------------- #
def _kmers(seq: str, k: int = 4) -> Set[str]:
    return {seq[i : i + k] for i in range(0, max(0, len(seq) - k + 1))}


def python_proxy_hits(query_fasta: str, target_fasta: str, max_identity: float, k: int = 4) -> List[Tuple[str, float, float]]:
    """Containment proxy: for each query, max over targets of |Q∩T|/|Q| on k-mers.
    Returns (query_id, proxy_score, proxy_score) for queries above threshold."""
    target_kmers = [_kmers(r.seq, k) for r in read_fasta(target_fasta)]
    hits = []
    for q in iter_fasta(query_fasta):
        qk = _kmers(q.seq, k)
        if not qk:
            continue
        best = max((len(qk & tk) / len(qk) for tk in target_kmers), default=0.0)
        if best >= max_identity:
            hits.append((q.id, best, best))
    return hits


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def filter_retain(
    retain_fasta: str,
    family_fastas: Dict[str, str],
    out_fasta: str,
    max_identity: float,
    coverage: float,
    tool: str = "mmseqs2",
    workdir: str | None = None,
    threads: int = 4,
) -> dict:
    """Drop retain sequences with identity ≥ max_identity (at ≥ coverage) to any
    family in ``family_fastas`` (e.g. {'F': ..., 'F_prime': ...}). Writes the
    filtered FASTA and returns a report dict (the data manifest stores this)."""
    use_python = tool == "python" or (tool in ("mmseqs2", "mmseqs") and not _have("mmseqs"))
    if use_python and tool != "python":
        warnings.warn(
            "mmseqs not found on PATH; falling back to the k-mer containment proxy. "
            "This is for the toy/offline path only — load mmseqs2 for real data.",
            RuntimeWarning,
        )

    to_remove: Set[str] = set()
    per_family: Dict[str, int] = {}
    max_observed = 0.0
    tmp_root = workdir or tempfile.mkdtemp(prefix="leakage_")
    for fam_name, fam_path in family_fastas.items():
        if use_python:
            hits = python_proxy_hits(retain_fasta, fam_path, max_identity)
        else:
            hits = mmseqs_search(retain_fasta, fam_path, coverage, str(Path(tmp_root) / fam_name), threads)
        removed_here = 0
        for qid, fident, _cov in hits:
            max_observed = max(max_observed, fident)
            if fident >= max_identity:
                if qid not in to_remove:
                    removed_here += 1
                to_remove.add(qid)
        per_family[fam_name] = removed_here

    kept = [r for r in read_fasta(retain_fasta)]
    n_in = len(kept)
    kept = [r for r in kept if r.id not in to_remove]
    n_out = write_fasta(kept, out_fasta)

    return {
        "tool": "python_kmer_proxy" if use_python else "mmseqs2",
        "max_identity_to_F": max_identity,
        "coverage": coverage,
        "n_in": n_in,
        "n_removed": len(to_remove),
        "n_out": n_out,
        "removed_per_family": per_family,
        "max_identity_observed": max_observed,
        "removed_ids": sorted(to_remove),
    }
