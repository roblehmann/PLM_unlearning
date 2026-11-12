"""Datamodule: canonical on-disk layout, train/test splits, budget subsampling,
attacker-pool construction, and MLM dataloaders.

On-disk layout under ``<paths.data>``::

    filtered/retain.filtered.fasta
    splits/{F_train,F_test,F_prime_train,F_prime_test,retain_holdout,background}.fasta
    homologs/{F,F_prime}_homologs.fasta
    raw/{F,F_prime}/family.hmm
    manifest.json, leakage_report.json
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from src.utils.io import SeqRecord, fasta_seqs, load_json, read_fasta


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass
class DataPaths:
    root: str

    @property
    def base(self) -> Path:
        return Path(self.root)

    # filtered retain
    @property
    def retain_filtered(self) -> Path:
        return self.base / "filtered" / "retain.filtered.fasta"

    # splits
    def split(self, name: str) -> Path:
        return self.base / "splits" / f"{name}.fasta"

    @property
    def F_train(self):  return self.split("F_train")
    @property
    def F_test(self):   return self.split("F_test")
    @property
    def Fp_train(self): return self.split("F_prime_train")
    @property
    def Fp_test(self):  return self.split("F_prime_test")
    @property
    def retain_train(self):   return self.split("retain_train")
    @property
    def retain_holdout(self): return self.split("retain_holdout")
    @property
    def background(self):     return self.split("background")

    # homologs
    def homologs(self, family: str) -> Path:
        return self.base / "homologs" / f"{family}_homologs.fasta"

    # hmm
    def hmm(self, family: str) -> Path:
        sub = "F" if family in ("F", "F_train", "F_test") else "F_prime"
        return self.base / "raw" / sub / "family.hmm"

    @property
    def manifest(self):        return self.base / "manifest.json"
    @property
    def leakage_report(self):  return self.base / "leakage_report.json"

    def load_manifest(self) -> dict:
        return load_json(self.manifest)


# --------------------------------------------------------------------------- #
# Splits / subsampling
# --------------------------------------------------------------------------- #
def random_split(records: Sequence[SeqRecord], train_frac: float, seed: int = 0) -> Tuple[List[SeqRecord], List[SeqRecord]]:
    recs = list(records)
    rng = random.Random(seed)
    rng.shuffle(recs)
    k = int(round(len(recs) * train_frac))
    return recs[:k], recs[k:]


def subsample(sequences: Sequence[str], n: int, seed: int = 0) -> List[str]:
    """Return min(n, len) sequences sampled without replacement (deterministic)."""
    seqs = list(sequences)
    rng = random.Random(seed)
    rng.shuffle(seqs)
    return seqs[: min(n, len(seqs))]


# --------------------------------------------------------------------------- #
# Attacker pool / target selection (used by the sweep)
# --------------------------------------------------------------------------- #
def attacker_pool(paths: DataPaths, target_family: str, adversary_data: str) -> List[str]:
    """Sequences the attacker may draw its budget from.

    target_family: 'F' or 'F_prime'.  adversary_data: 'clean' | 'homolog_augmented' |
    'retain_only'.  For 'retain_only' the pool is the retain set R (zero target-family
    sequences); ``target_family`` then selects only the family whose held-out PPL is
    scored, not the training pool.
    """
    if adversary_data == "retain_only":
        return fasta_seqs(paths.retain_train)
    train = paths.F_train if target_family == "F" else paths.Fp_train
    pool = fasta_seqs(train)
    if adversary_data == "homolog_augmented":
        fam = "F" if target_family == "F" else "F_prime"
        hpath = paths.homologs(fam)
        if Path(hpath).exists():
            pool = pool + fasta_seqs(hpath)
    return pool


def target_test_seqs(paths: DataPaths, target_family: str) -> List[str]:
    return fasta_seqs(paths.F_test if target_family == "F" else paths.Fp_test)


def attacker_trainset(
    paths: DataPaths,
    target_family: str,
    adversary_data: str,
    budget_n: int,
    seed: int = 0,
    homolog_ratio: float = 1.0,
) -> List[str]:
    """Build the attacker's training set for one sweep cell.

    clean             -> ``budget_n`` real target-family train sequences.
    homolog_augmented -> the SAME ``budget_n`` real sequences PLUS
                         round(homolog_ratio * budget_n) homologs added on top.
    retain_only       -> ``budget_n`` retain (R) sequences and ZERO target-family
                         examples — the retain-only *relearning* probe. The attacker
                         fine-tunes purely on the benign retain objective B was trained
                         to keep; it never sees F. Here ``target_family`` selects only
                         the held-out set whose pseudo-perplexity is *scored* (see
                         ``target_test_seqs``), NOT the training data — so
                         (retain_only, F) and (retain_only, F_prime) share an identical
                         training set and differ only in which family's PPL is watched.
                         If F-PPL collapses under R-only tuning, the forget capability
                         was dormant (suppressed), not deleted.

    For clean/homolog this is a strict *augmentation* (more data than clean at the
    same real-example budget), not a dilution of the budget — so the real examples
    still carry their signal and the multiplier stays well-defined.
    """
    if adversary_data == "retain_only":
        # The exact retain set the defender trained on (see scripts/03_train_blocked.py:
        # retain = fasta_seqs(paths.retain_train)); zero F. Swap to paths.retain_holdout
        # for the "fresh benign data the attacker happens to hold" variant.
        return subsample(fasta_seqs(paths.retain_train), budget_n, seed=seed)

    train = paths.F_train if target_family == "F" else paths.Fp_train
    real = subsample(fasta_seqs(train), budget_n, seed=seed)
    if adversary_data == "homolog_augmented" and homolog_ratio > 0:
        fam = "F" if target_family == "F" else "F_prime"
        hpath = paths.homologs(fam)
        if Path(hpath).exists():
            n_hom = int(round(homolog_ratio * budget_n))
            homs = subsample(fasta_seqs(hpath), n_hom, seed=seed + 99991)
            return list(real) + list(homs)
    return list(real)


# --------------------------------------------------------------------------- #
# MLM dataloaders
# --------------------------------------------------------------------------- #
class SequenceDataset:
    """Map-style dataset returning pre-tokenized input_ids/attention_mask. Not a
    torch.utils.data.Dataset subclass, so importing this module needs no torch."""

    def __init__(self, sequences: Sequence[str], tokenizer, max_length: int = 1024):
        self.sequences = list(sequences)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tok(self.sequences[idx], truncation=True, max_length=self.max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def make_mlm_loader(
    sequences: Sequence[str],
    tokenizer,
    batch_size: int,
    mlm_probability: float = 0.15,
    max_length: int = 1024,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = False,
):
    """Build a DataLoader that yields MLM batches (input_ids, attention_mask, labels)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
    ds = SequenceDataset(sequences, tokenizer, max_length=max_length)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        generator=g,
        drop_last=drop_last,
    )


def cycle(loader) -> Iterator:
    """Infinite iterator over a DataLoader (for step-based training loops)."""
    while True:
        for batch in loader:
            yield batch
