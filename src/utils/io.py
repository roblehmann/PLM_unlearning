"""Small IO helpers: a dependency-free FASTA reader/writer, JSON helpers, device
selection. Kept Biopython-free so tests run in a minimal environment."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Union

PathLike = Union[str, os.PathLike]


@dataclass
class SeqRecord:
    id: str
    seq: str


def read_fasta(path: PathLike) -> List[SeqRecord]:
    """Parse a FASTA file into SeqRecords. Sequence whitespace is stripped."""
    records: List[SeqRecord] = []
    header: str | None = None
    chunks: List[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(SeqRecord(header, "".join(chunks)))
                header = line[1:].split()[0]  # id = first token after '>'
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            records.append(SeqRecord(header, "".join(chunks)))
    return records


def iter_fasta(path: PathLike) -> Iterator[SeqRecord]:
    """Streaming FASTA reader for large files."""
    header: str | None = None
    chunks: List[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield SeqRecord(header, "".join(chunks))
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield SeqRecord(header, "".join(chunks))


def write_fasta(records: Iterable[Union[SeqRecord, Tuple[str, str]]], path: PathLike, width: int = 80) -> int:
    """Write records (SeqRecord or (id, seq) tuples) to FASTA. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for rec in records:
            rid, seq = (rec.id, rec.seq) if isinstance(rec, SeqRecord) else rec
            fh.write(f">{rid}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")
            n += 1
    return n


def fasta_seqs(path: PathLike) -> List[str]:
    return [r.seq for r in read_fasta(path)]


def count_fasta(path: PathLike) -> int:
    n = 0
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def save_json(obj, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def load_json(path: PathLike) -> Dict:
    with open(path, "r") as fh:
        return json.load(fh)


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(prefer: str = "cuda") -> str:
    """Return 'cuda' if requested and available, else 'cpu'. Never hard-fails."""
    if prefer.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return prefer
        except ImportError:
            pass
    return "cpu"
