"""Shared plumbing for the experiment-7 anchor screen.

The question this stage answers: which Pfam families sit FAR FROM THE BULK of Pfam
itself, in an isolated corner of ESM-2 embedding space, and are usable as the next
task-blocking anchor F (>=1000 members, not only very short sequences, not only
repetitive sequence, preferably outside a clan)?

`cluster_families.py` already ranks families by distance to the retain corpus R. That is
a different axis, and its average-linkage cut is degenerate: 20,833 of 21,763 families
land in one cluster whose median distance-to-retain is 0.0158. Mean-pooled ESM-2
embeddings are strongly anisotropic — a shared mean direction dominates every cosine — so
this stage measures that anisotropy, removes it by centering, and scores isolation with
density-aware statistics instead of one linkage cut.

Everything here is torch-free at import time (``src.*`` and torch are imported lazily
inside functions) so ``test_anchors.py`` runs offline without a GPU.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (plm_unlearning/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TASKBLOCK_ROOT", str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# A smoke run sets TB_ANCHOR_TAG so its artifacts land beside — never on top of — the real
# ones, mirroring exp-6's TB_MGREPS_TAG.
TAG = os.environ.get("TB_ANCHOR_TAG", "").strip()
_SUFFIX = f"_{TAG}" if TAG else ""
RESULTS = HERE / f"results{_SUFFIX}" / "anchors"

# --------------------------------------------------------------------------- #
# Inputs — all produced by earlier stages of this experiment; never recomputed here.
# --------------------------------------------------------------------------- #
EMBED_DIR = HERE / "results" / "embeddings"      # always the REAL embeddings, even in smoke
CLUSTERING_DIR = HERE / "results" / "clustering"
UMAP_DIR = HERE / "results" / "umap"
META_TSV = ROOT / "data/Pfam_family_representatives/family_representatives.tsv"
REPS_FASTA = ROOT / "data/Pfam_family_representatives/family_representatives.fasta"
RETAIN_FASTA = ROOT / "data/splits/retain_train.fasta"
LADDER_DIR = ROOT / "data/ladder"
DATA_CONFIG = ROOT / "configs/data/pfam_default.yaml"

MODEL_ID = "facebook/esm2_t12_35M_UR50D"
HIDDEN_DIM = 480
MAX_RESIDUES = 1022
EMBED_BATCH = 128
SEED = 0

N_REPS_EXPECTED = 21_763
N_RETAIN_EXPECTED = 194_064

# Cached F-set embeddings (train halves) — used only for `max_cos_to_existing_anchor`.
EXISTING_ANCHOR_GROUPS = ("MGYF0000018690_F_train", "PF00106_F_train", "PF00107_F_train")

# Families this project already uses. Read from disk by `load_project_families`; this is
# the fallback/expectation used by the tests and by the provenance block.
CONFIG_FAMILIES = ("PF00106", "PF00107", "PF01593", "PF01370")
PROJECT_CLAN = "CL0063"          # NADP_Rossmann — the clan both current anchors come from

SPACES = ("raw", "cen", "whi")
HEADLINE_SPACE = "cen"

# Isolation components averaged into the headline score (equal weights by default).
ISOLATION_COMPONENTS = (
    "iso_nn_xclan", "iso_top10_xclan", "lof50",
    "iso_centroid", "first_merge_height", "height_join20",
)

# Gate order is fixed so the waterfall in t3 is meaningful and reproducible.
GATE_ORDER = ("gate_size", "gate_length", "gate_complexity",
              "gate_hazard", "gate_project", "gate_isolation")

# Repetitiveness / low-complexity metrics. `higher_is_worse=False` means a LOW value is
# the suspicious one (zlib_ratio: repetitive text compresses well).
@dataclass(frozen=True)
class SeqMetric:
    name: str
    higher_is_worse: bool


SEQ_METRICS: Tuple[SeqMetric, ...] = (
    SeqMetric("lowcomp_frac_w12", True),
    SeqMetric("max_aa_frac", True),
    SeqMetric("kmer_dup_frac_k3", True),
    SeqMetric("max_periodicity", True),
    SeqMetric("zlib_ratio", False),
)

LOWCOMP_WINDOW = 12
LOWCOMP_BITS = 2.2               # SEG-*like*; the flag threshold itself is calibrated
MAX_PERIOD = 50
RETAIN_LENGTH_DECILES = 10

STD_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {a: i for i, a in enumerate(STD_AA)}
N_CODES = len(STD_AA) + 1        # +1 bucket for X/B/Z/U/O and anything else

# Okabe-Ito, same hues as the existing exp-7 figures.
C_CANDIDATE = "#D55E00"
C_PROJECT = "#0072B2"
C_BULK = "#9ba5ad"
C_REPEAT = "#CC79A7"
C_OK = "#009E73"
SEQ_CMAP = "cividis"
MATRIX_CMAP = "magma_r"


# --------------------------------------------------------------------------- #
# TSV / hashing / normalisation — lifted verbatim from cluster_families.py so both
# scripts share one implementation.
# --------------------------------------------------------------------------- #
def read_tsv(path) -> List[dict]:
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(path, rows: Sequence[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return path
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'  wrote {path} ({len(rows):,} rows)', flush=True)
    return path


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def normalize(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms <= 0):
        raise ValueError('Non-finite or zero embedding')
    return x / norms


def neighbors(query, reference, k=10, exclude=None):
    """Exact cosine similarities, bounded memory, BLAS matrix multiplication.

    Verbatim from cluster_families.py: both stages must produce the same numbers for the
    retain search. Returns (best, mean_top_k, argmax_index) per query row.
    """
    best, mean, ids = [], [], []
    for start in range(0, len(query), 256):
        sim = query[start:start+256] @ reference.T
        np.clip(sim, -1, 1, out=sim)
        if exclude is not None:
            for i, indices in enumerate(exclude[start:start+len(sim)]):
                sim[i, indices] = -np.inf
        idx = sim.argmax(axis=1)
        top = np.partition(sim, sim.shape[1]-k, axis=1)[:, -k:]
        best.extend(sim[np.arange(len(sim)), idx].tolist())
        mean.extend(top.mean(axis=1).tolist())
        ids.extend(idx.tolist())
        if start % 2560 == 0:
            print(f'Exact neighbor search: {start:,}/{len(query):,}', flush=True)
    return np.array(best), np.array(mean), np.array(ids)


# --------------------------------------------------------------------------- #
# Cached embeddings
# --------------------------------------------------------------------------- #
def load_group(group: str, verify: bool = True) -> Tuple[np.ndarray, List[dict]]:
    """Load one cached embedding group, checking it against `manifest.json` first.

    Same contract as cluster_families.py: a checksum mismatch is fatal, because every
    number downstream is keyed to these exact bytes.
    """
    manifest = json.loads((EMBED_DIR / 'manifest.json').read_text())
    if group not in manifest['groups']:
        raise SystemExit(f'{group} is not in {EMBED_DIR}/manifest.json')
    if verify:
        for suffix, key in (('.npy', 'embeddings_sha256'), ('.tsv', 'metadata_sha256')):
            path = EMBED_DIR / f'{group}{suffix}'
            if sha(path) != manifest['groups'][group][key]:
                raise SystemExit(f'Input checksum mismatch: {path}')
    rows = read_tsv(EMBED_DIR / f'{group}.tsv')
    x = np.load(EMBED_DIR / f'{group}.npy')
    if x.shape != (len(rows), HIDDEN_DIM):
        raise SystemExit(f'{group}: embeddings {x.shape} do not match {len(rows)} metadata rows')
    return x, rows


def load_metadata() -> Dict[str, dict]:
    rows = read_tsv(META_TSV)
    table = {r['pfam_acc']: r for r in rows}
    if len(table) != len(rows):
        raise SystemExit(f'Duplicate Pfam accessions in {META_TSV}')
    return table


def load_reps(limit: int = 0, verify: bool = True):
    """(embeddings, per-row embedding metadata, joined Pfam metadata) in FASTA order.

    ``limit`` takes a deterministic stride over the reps (not a prefix) so a smoke run
    still spans the whole accession range and the clan/type mixture stays realistic.
    """
    x, rows = load_group('pfam_reps', verify=verify)
    table = load_metadata()
    missing = [r['id'] for r in rows if r['id'] not in table]
    if missing:
        raise SystemExit(f'{len(missing)} representatives have no metadata row, e.g. {missing[:3]}')
    if limit and limit < len(rows):
        keep = np.linspace(0, len(rows) - 1, int(limit)).astype(int)
        keep = np.unique(keep)
        x, rows = x[keep], [rows[i] for i in keep]
    meta = [table[r['id']] for r in rows]
    return x, rows, meta


def load_project_families() -> Dict[str, str]:
    """Every Pfam accession this project already uses -> why it is in use.

    Read from disk (the two ladder JSONs + the data config) rather than hardcoded, so a
    new ladder cannot silently reintroduce a family the screen is meant to avoid.
    """
    used: Dict[str, str] = {}
    for acc in CONFIG_FAMILIES:
        used[acc] = 'configs/data/pfam_default.yaml'
    for path in sorted(LADDER_DIR.glob('*/ladder_selected.json')):
        payload = json.loads(path.read_text())          # loud on a corrupt ladder file
        anchor = payload.get('anchor_F', path.parent.name)
        used.setdefault(str(anchor), f'{path.relative_to(ROOT)} (anchor)')
        for rung in payload.get('rungs', []):
            acc = rung.get('acc')
            if acc:
                used[str(acc)] = f'{path.relative_to(ROOT)} (rung {rung.get("rung")})'
    return used


# --------------------------------------------------------------------------- #
# Embedding spaces
# --------------------------------------------------------------------------- #
@dataclass
class Space:
    """A fitted transform applied to embeddings before any cosine is taken.

    `raw` reproduces the existing run; `cen` is the headline (the shared mean direction
    that compresses every cosine towards 1 is removed); `whi` is a sensitivity only —
    whitening amplifies low-variance directions, which in a 35M model are partly noise.
    """
    key: str
    mean: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None       # (k, d)
    scale: Optional[np.ndarray] = None            # (k,)
    eigenvalues: Optional[np.ndarray] = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.mean is not None:
            x = x - self.mean
        if self.components is not None:
            x = (x @ self.components.T) * self.scale
        return normalize(x)

    def as_dict(self) -> dict:
        d = {'space': self.key}
        if self.components is not None:
            d['n_components'] = int(self.components.shape[0])
        return d

    def save(self, path) -> Path:
        arrays = {k: v for k, v in (('mean', self.mean), ('components', self.components),
                                    ('scale', self.scale), ('eigenvalues', self.eigenvalues))
                  if v is not None}
        np.savez(path, key=np.array(self.key), **arrays)
        return Path(path)

    @staticmethod
    def load(path) -> 'Space':
        z = np.load(path, allow_pickle=False)
        return Space(key=str(z['key']),
                     mean=z['mean'] if 'mean' in z else None,
                     components=z['components'] if 'components' in z else None,
                     scale=z['scale'] if 'scale' in z else None,
                     eigenvalues=z['eigenvalues'] if 'eigenvalues' in z else None)


def fit_space(x: np.ndarray, key: str, n_components: int = 128, eps: float = 1e-3,
              seed: int = SEED) -> Space:
    """Fit a transform on the representatives only; retain is projected through it."""
    x = np.asarray(x, dtype=np.float32)
    if key == 'raw':
        return Space('raw')
    mean = x.mean(axis=0)
    if key == 'cen':
        return Space('cen', mean=mean)
    if key == 'whi':
        from sklearn.decomposition import PCA

        k = int(min(n_components, x.shape[0] - 1, x.shape[1]))
        pca = PCA(n_components=k, svd_solver='randomized', random_state=int(seed))
        pca.fit(x - mean)
        lam = np.asarray(pca.explained_variance_, dtype=np.float32)
        scale = (1.0 / np.sqrt(lam + eps * float(lam.mean()))).astype(np.float32)
        return Space('whi', mean=mean, components=pca.components_.astype(np.float32),
                     scale=scale, eigenvalues=lam)
    raise ValueError(f'Unknown space: {key}')


def anisotropy(unit: np.ndarray) -> dict:
    """How much of the geometry is one shared direction.

    ``||mean of the L2-normalised rows||`` is the number: 1.0 means every vector points the
    same way (every cosine is ~1 and no isolation statistic can mean anything), 0.0 means
    the directions cancel. Reported per space beside the effective rank of the covariance.
    """
    unit = np.asarray(unit, dtype=np.float64)
    mean_vec = unit.mean(axis=0)
    cov = np.cov(unit, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)[::-1]
    total = float(lam.sum())
    p = lam / total if total > 0 else lam
    nz = p[p > 0]
    return {
        'mean_resultant_length': float(np.linalg.norm(mean_vec)),
        'effective_rank': float(np.exp(-np.sum(nz * np.log(nz)))) if nz.size else float('nan'),
        'participation_ratio': float(total ** 2 / float(np.sum(lam ** 2))) if total > 0 else float('nan'),
        'top_eigenvalue_frac': float(p[0]) if p.size else float('nan'),
        'n_dim': int(unit.shape[1]),
    }


# --------------------------------------------------------------------------- #
# Ranking primitives
# --------------------------------------------------------------------------- #
def pctrank(x: Sequence[float]) -> np.ndarray:
    """Percentile rank in [0, 100] with ties averaged (scipy rankdata convention)."""
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    return rankdata(x, method='average') / x.size * 100.0


def robust_z(x: Sequence[float]) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); zeros when the MAD vanishes."""
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 0:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def lof_from_topk(dist: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """Local outlier factor from precomputed k-nearest-neighbour arrays.

    ``dist``/``idx`` are (n, K>=k) neighbour distances/indices sorted ascending, self
    excluded. LOF > 1 means the point is in a sparser neighbourhood than its own
    neighbours are — which is exactly "isolated location" as opposed to "far from the
    centre". Computed from the (n, K) arrays because sklearn's precomputed path would
    want a float64 n^2 matrix: 3.8 GB at n=21,763, for no extra information.
    """
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx)
    k = int(k)
    if k < 1 or k > dist.shape[1]:
        raise ValueError(f'k={k} outside the stored neighbourhood of {dist.shape[1]}')
    kdist = dist[:, k - 1]
    reach = np.maximum(kdist[idx[:, :k]], dist[:, :k])
    lrd = 1.0 / np.maximum(reach.mean(axis=1), 1e-12)
    return lrd[idx[:, :k]].mean(axis=1) / lrd


def linkage_leaf_heights(z: np.ndarray, n: int,
                         thresholds: Sequence[int] = (20, 100)) -> Dict[str, np.ndarray]:
    """Per-leaf merge heights from a scipy linkage matrix.

    * ``first_merge_height`` — the height at which a family stops being alone. A family
      whose nearest relative is far away merges late.
    * ``height_join<m>`` — the height at which its cluster first reaches m leaves, i.e.
      how far you must travel before it is absorbed into something bulk-sized. This is
      the metric that survives the degenerate 95.7%-cluster problem, because it does not
      depend on where a global cut is placed.
    """
    z = np.asarray(z, dtype=np.float64)
    n = int(n)
    if z.shape != (n - 1, 4):
        raise ValueError(f'linkage {z.shape} does not describe {n} leaves')
    size = np.ones(2 * n - 1, dtype=np.int64)
    first = np.full(n, np.nan)
    order = np.full(n, np.nan)
    for i, (a, b, h, _) in enumerate(z):
        a, b = int(a), int(b)
        node = n + i
        size[node] = size[a] + size[b]
        for child in (a, b):
            if child < n:
                first[child] = h
                order[child] = (i + 1) / (n - 1)
    out = {'first_merge_height': first, 'merge_order_frac': order}
    for m in thresholds:
        # Small-to-large merging of the still-unassigned leaves: O(n log n) overall.
        pending: Dict[int, List[int]] = {i: [i] for i in range(n)}
        height = np.full(n, np.nan)
        for i, (a, b, h, _) in enumerate(z):
            a, b, node = int(a), int(b), n + i
            left, right = pending.pop(a, []), pending.pop(b, [])
            if len(left) < len(right):
                left, right = right, left
            left.extend(right)
            if size[node] >= int(m):
                for leaf in left:
                    height[leaf] = h
                left = []
            pending[node] = left
        out[f'height_join{int(m)}'] = height
    return out


# --------------------------------------------------------------------------- #
# Sequence quality — length, low complexity, repetitiveness
# --------------------------------------------------------------------------- #
def domain_spans(ranges: str, length: Optional[int] = None) -> List[Tuple[int, int]]:
    """Parse Pfam ``rep_domain_ranges`` ("58-425", "21-81,97-166") to 0-based half-open
    spans. Never raises: a malformed or empty field yields [] and the caller decides."""
    spans: List[Tuple[int, int]] = []
    for chunk in str(ranges or '').replace(';', ',').split(','):
        chunk = chunk.strip()
        if not chunk or '-' not in chunk:
            continue
        lo, _, hi = chunk.partition('-')
        try:
            start, end = int(lo), int(hi)
        except ValueError:
            continue
        if start < 1 or end < start:
            continue
        start -= 1                                   # Pfam is 1-based inclusive
        if length is not None:
            start, end = max(0, min(start, length)), max(0, min(end, length))
        if end > start:
            spans.append((start, end))
    return sorted(spans)


def domain_sequence(seq: str, ranges: str) -> str:
    spans = domain_spans(ranges, len(seq))
    if not spans:
        return seq
    return ''.join(seq[a:b] for a, b in spans)


def encode(seq: str) -> np.ndarray:
    codes = np.full(len(seq), len(STD_AA), dtype=np.int8)
    for i, ch in enumerate(seq.upper()):
        j = _AA_INDEX.get(ch)
        if j is not None:
            codes[i] = j
    return codes


def window_entropy(codes: np.ndarray, window: int = LOWCOMP_WINDOW) -> np.ndarray:
    """Shannon entropy (bits) of every ``window``-wide window, via a one-hot cumsum."""
    codes = np.asarray(codes)
    L = codes.size
    w = int(window)
    if L < w or w < 1:
        return np.empty(0, dtype=np.float64)
    onehot = np.zeros((L, N_CODES), dtype=np.float64)
    onehot[np.arange(L), codes] = 1.0
    cum = np.vstack([np.zeros((1, N_CODES)), np.cumsum(onehot, axis=0)])
    counts = cum[w:] - cum[:-w]
    p = counts / float(w)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=1)


def lowcomp_fraction(codes: np.ndarray, window: int = LOWCOMP_WINDOW,
                     bits: float = LOWCOMP_BITS) -> float:
    """Fraction of residues covered by at least one low-entropy window (SEG-*like*).

    Not SEG: no optimal-segment extension, and the decision threshold used downstream is
    calibrated against retain rather than taken from the SEG defaults.
    """
    codes = np.asarray(codes)
    L = codes.size
    if L == 0:
        return float('nan')
    ent = window_entropy(codes, window)
    if ent.size == 0:                                # sequence shorter than one window
        whole = window_entropy(codes, L)
        return float(whole[0] < bits) if whole.size else float('nan')
    low = np.flatnonzero(ent < bits)
    if low.size == 0:
        return 0.0
    diff = np.zeros(L + 1, dtype=np.int32)
    np.add.at(diff, low, 1)
    np.add.at(diff, low + int(window), -1)
    return float((np.cumsum(diff[:L]) > 0).mean())


def max_periodicity(codes: np.ndarray, max_period: int = MAX_PERIOD) -> Tuple[float, int]:
    """Best tandem-repeat signal: max over periods p of the fraction of positions with
    ``seq[i] == seq[i+p]``. A perfect (ABCDE)xN repeat scores 1.0 at p=5."""
    codes = np.asarray(codes)
    L = codes.size
    if L < 4:
        return float('nan'), 0
    best, best_p = 0.0, 0
    for p in range(1, int(min(max_period, L // 2)) + 1):
        score = float(np.mean(codes[:-p] == codes[p:]))
        if score > best:
            best, best_p = score, p
    return best, best_p


def seq_stats(seq: str) -> dict:
    """Every length / low-complexity / repetitiveness statistic for one sequence."""
    seq = str(seq).upper()
    L = len(seq)
    if L == 0:
        return {'length': 0, 'aa_entropy_bits': float('nan'), 'lowcomp_frac_w12': float('nan'),
                'max_aa_frac': float('nan'), 'longest_homopolymer': 0,
                'kmer_dup_frac_k3': float('nan'), 'top_kmer_frac_k3': float('nan'),
                'max_periodicity': float('nan'), 'best_period': 0, 'zlib_ratio': float('nan')}
    codes = encode(seq)
    counts = np.bincount(codes, minlength=N_CODES).astype(np.float64)
    p = counts[counts > 0] / L
    runs = np.diff(np.flatnonzero(np.concatenate(([1], np.diff(codes) != 0, [1]))))
    if L >= 3:
        k3 = (codes[:-2].astype(np.int64) * N_CODES + codes[1:-1]) * N_CODES + codes[2:]
        uniq, k3counts = np.unique(k3, return_counts=True)
        dup = 1.0 - uniq.size / float(k3.size)
        top = float(k3counts.max()) / float(k3.size)
    else:
        dup, top = float('nan'), float('nan')
    period, best_p = max_periodicity(codes)
    return {
        'length': int(L),
        'aa_entropy_bits': float(-np.sum(p * np.log2(p))),
        'lowcomp_frac_w12': lowcomp_fraction(codes),
        'max_aa_frac': float(counts.max() / L),
        'longest_homopolymer': int(runs.max()) if runs.size else 0,
        'kmer_dup_frac_k3': float(dup),
        'top_kmer_frac_k3': float(top),
        'max_periodicity': float(period),
        'best_period': int(best_p),
        'zlib_ratio': float(len(zlib.compress(seq.encode(), 9)) / L),
    }


def length_decile_edges(lengths: Sequence[int], n_bins: int = RETAIN_LENGTH_DECILES) -> np.ndarray:
    """Interior quantile edges of the reference length distribution."""
    q = np.linspace(0, 100, int(n_bins) + 1)[1:-1]
    return np.quantile(np.asarray(lengths, dtype=np.float64), q / 100.0)


def length_bin(length: int, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(length), side='right'))


def percentile_against(value: float, sorted_reference: np.ndarray) -> float:
    """Percentile of ``value`` within an ascending reference sample, in [0, 100]."""
    ref = np.asarray(sorted_reference, dtype=np.float64)
    if ref.size == 0 or not np.isfinite(value):
        return float('nan')
    return float(np.searchsorted(ref, float(value), side='right') / ref.size * 100.0)


# --------------------------------------------------------------------------- #
# Embedding (torch imported lazily)
# --------------------------------------------------------------------------- #
def embed_length_sorted(model, tokenizer, seqs: List[str], device: str, batch_size: int,
                        max_length: int = 1024) -> np.ndarray:
    """``embed_sequences`` with length bucketing, then the permutation undone.

    Copied from scripts/12_embed_corpora.py — the module name starts with a digit and
    cannot be imported, and exp-5/exp-6's common.py set the precedent of copying rather
    than importlib-ing. Byte-identical pooling to the cached representative embeddings.
    """
    import torch

    from src.eval.membership import embed_sequences

    order = np.argsort([len(s) for s in seqs], kind='stable')
    ordered = [seqs[i] for i in order]
    if device.startswith('cuda'):
        with torch.autocast('cuda', dtype=torch.float16):
            feats = embed_sequences(model, tokenizer, ordered, device=device,
                                    batch_size=batch_size, max_length=max_length)
    else:
        feats = embed_sequences(model, tokenizer, ordered, device=device,
                                batch_size=batch_size, max_length=max_length)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(feats)[inverse]


# --------------------------------------------------------------------------- #
# IO / provenance / plotting
# --------------------------------------------------------------------------- #
def results_dir(*parts: str) -> Path:
    d = RESULTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def apply_results_override(path: Optional[str]) -> None:
    global RESULTS
    if path:
        RESULTS = Path(path)
    RESULTS.mkdir(parents=True, exist_ok=True)


def _denan(o):
    if isinstance(o, dict):
        return {k: _denan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_denan(v) for v in o]
    if isinstance(o, np.generic):
        return _denan(o.item())
    if isinstance(o, np.ndarray):
        return [_denan(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_denan(obj), indent=2) + '\n')
    print(f'  wrote {path}', flush=True)
    return path


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


def banner(title: str) -> None:
    print('\n' + '=' * 78 + f'\n{title}\n' + '=' * 78, flush=True)


def versions(packages: Iterable[str] = ('numpy', 'scipy', 'scikit-learn', 'matplotlib')) -> dict:
    import importlib.metadata

    out = {}
    for p in packages:
        try:
            out[p] = importlib.metadata.version(p)
        except Exception:                                  # noqa: BLE001 — provenance only
            out[p] = 'n/a'
    return out


def protocol_block(**overrides) -> dict:
    block = {
        'experiment': '7_pfam_reps_vs_retain (anchor screen)',
        'model_id': MODEL_ID,
        'embeddings': str(EMBED_DIR.relative_to(ROOT)),
        'space_headline': HEADLINE_SPACE,
        'metadata': str(META_TSV.relative_to(ROOT)),
        'tag': TAG,
        'seed': SEED,
        'versions': versions(),
    }
    block.update(overrides)
    return block


def common_args(parser):
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--results', default=None,
                        help='override the results dir (default: results/anchors or results_<TAG>/anchors)')
    parser.add_argument('--no-verify', action='store_true',
                        help='skip input checksum verification (smoke only)')
    return parser


def guard_limits(args) -> None:
    """Refuse a subsampled run that would write into the real results directory.

    Same guard the existing exp-7 scripts apply to `--limit`: a partial run must never
    overwrite full-corpus artifacts, because nothing downstream can tell them apart.
    """
    limited = any(getattr(args, name, 0) for name in ('limit_families', 'limit_retain'))
    if limited and not TAG and not getattr(args, 'results', None):
        raise SystemExit('--limit-* requires TB_ANCHOR_TAG=<tag> or --results <dir>; '
                         'the default results directory takes full runs only.')


def set_rc() -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'figure.dpi': 110, 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9,
        'legend.frameon': False,
    })


def style_axis(ax, ygrid: bool = True) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ygrid:
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)


def save_fig(fig, stem: str, dpi: int = 200) -> List[Path]:
    out = results_dir('plots')
    paths = []
    for ext in ('png', 'pdf'):
        p = out / f'{stem}.{ext}'
        fig.savefig(p, dpi=dpi)
        paths.append(p)
        print(f'  wrote {p}', flush=True)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths
