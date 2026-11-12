"""Homolog generation for the durability stressor (Phase 6, adversary_data=homolog_augmented).

Primary: `hmmemit` samples synthetic sequences from the family profile HMM — these
are the "obvious circumvention" the durability axis is meant to test. Fallback (no
HMMER, or a toy placeholder HMM): mutate the seed sequences at a fixed rate.
"""
from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path
from typing import List

from src.utils.io import SeqRecord, read_fasta, write_fasta

AA = "ACDEFGHIKLMNPQRSTVWY"


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def _is_real_hmm(hmm_path: str) -> bool:
    try:
        with open(hmm_path) as fh:
            head = fh.readline()
        return head.startswith("HMMER")
    except OSError:
        return False


def emit_from_hmm(hmm_path: str, n: int, out_fasta: str, seed: int = 0) -> int:
    """`hmmemit -N n` — sample n sequences from the profile HMM."""
    cmd = ["hmmemit", "-N", str(n), "--seed", str(seed), str(hmm_path)]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # hmmemit writes FASTA to stdout
    Path(out_fasta).parent.mkdir(parents=True, exist_ok=True)
    with open(out_fasta, "w") as fh:
        fh.write(proc.stdout)
    return sum(1 for line in proc.stdout.splitlines() if line.startswith(">"))


def mutate_seeds(seed_fasta: str, n_per_seed_seq: int, out_fasta: str, mutation_rate: float = 0.1, seed: int = 0) -> int:
    """Fallback: make ``n_per_seed_seq`` point-mutated copies of each seed sequence."""
    import random

    rng = random.Random(seed)
    seeds = read_fasta(seed_fasta)
    out: List[SeqRecord] = []
    for rec in seeds:
        for j in range(n_per_seed_seq):
            s = list(rec.seq)
            for i in range(len(s)):
                if rng.random() < mutation_rate:
                    s[i] = rng.choice(AA)
            out.append(SeqRecord(f"{rec.id}_hom{j}", "".join(s)))
    return write_fasta(out, out_fasta)


def generate_homologs(
    seed_fasta: str,
    hmm_path: str,
    n_per_seed_seq: int,
    out_fasta: str,
    tool: str = "hmmemit",
    seed: int = 0,
    mutation_rate: float = 0.1,
) -> int:
    """Generate homologs. Total count ≈ n_per_seed_seq × (#seed sequences). Uses
    hmmemit when available with a real HMM, else mutates the seeds."""
    n_seeds = sum(1 for _ in read_fasta(seed_fasta))
    n_total = n_per_seed_seq * max(1, n_seeds)
    use_hmm = tool == "hmmemit" and _have("hmmemit") and _is_real_hmm(hmm_path)
    if tool == "hmmemit" and not use_hmm:
        warnings.warn(
            "hmmemit unavailable or HMM is a toy placeholder; mutating seed sequences instead.",
            RuntimeWarning,
        )
    if use_hmm:
        return emit_from_hmm(hmm_path, n_total, out_fasta, seed=seed)
    return mutate_seeds(seed_fasta, n_per_seed_seq, out_fasta, mutation_rate=mutation_rate, seed=seed)
