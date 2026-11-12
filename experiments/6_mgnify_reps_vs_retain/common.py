"""Shared plumbing for experiment 6 — MGnify family representatives vs. the retain corpus.

The question: are the ~35k MGnify family representatives *novel* to ESM-2 in a way the
UniRef50 retain corpus is not? Two readouts, one model (base ESM-2 35M, theta_0 — no
task-blocked checkpoint is loaded anywhere in this experiment):

  * geometry      — one joint PCA-50 + UMAP over both corpora, plus the kNN-mixing number
                    that says quantitatively what the picture only suggests;
  * modelling     — per-sequence pseudo-perplexity, compared under length matching,
                    direct standardization and regression adjustment.

Everything here is torch-free at import time (``src.*`` and torch are imported lazily
inside functions) so ``test_mgreps.py`` can exercise the decision logic offline.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (taskblock-poc/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TASKBLOCK_ROOT", str(ROOT))

# Compute nodes have no egress. The 35M snapshot is already in the user cache, so pin the
# hub to offline BEFORE transformers is imported anywhere — otherwise a hub lookup blocks
# for its full timeout on an allocated GPU. t0 asserts the snapshot dir exists, so a cache
# miss fails in seconds on CPU rather than after a GPU allocation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# A smoke/scratch run sets TB_MGREPS_TAG so its artifacts land beside — never on top of —
# the real ones.
TAG = os.environ.get("TB_MGREPS_TAG", "").strip()
_SUFFIX = f"_{TAG}" if TAG else ""
RESULTS = HERE / f"results{_SUFFIX}"

# --------------------------------------------------------------------------- #
# Protocol — mirrors configs/config.yaml `eval:` EXACTLY.
# --------------------------------------------------------------------------- #
# Any deviation would make these pseudo-perplexities incomparable with tau, the headroom
# gate and every published PPL number in the project.
MODEL_ID = "facebook/esm2_t12_35M_UR50D"
HIDDEN_DIM = 480
MAX_LENGTH = 1024                 # => at most 1022 scored (non-special) positions
SCORED_MAX = MAX_LENGTH - 2
EVAL_BATCH = 64                   # configs/config.yaml eval.batch_size
EMBED_BATCH = 128                 # scripts/12_embed_corpora.py default
MASKS_PER_PASS = 8                # configs/config.yaml eval.masks_per_pass
SEED = 0
N_BOOT = 2000

# Per-forward token cap used to derive a safe batch size at long lengths. Numerically
# neutral: `_score_one` batches equal-length masked variants of ONE sequence (no padding,
# no cross-sequence interaction), so batch size only caps peak activation memory.
TOKEN_BUDGET = 131_072

# --------------------------------------------------------------------------- #
# Corpora
# --------------------------------------------------------------------------- #
REPS_FASTA = ROOT / "data/MGnify/family_representative_seq_corpus/mgnifams_reps.fasta"
RETAIN_FASTA = ROOT / "data/splits/retain_train.fasta"

# Base-model retain embeddings already computed by scripts/12_embed_corpora.py (full
# corpus, cap=-1, same model / max_length / pooling). Rows are in retain_train.fasta
# order; t0 verifies that id-for-id and t1 re-embeds a sample to confirm the numerical
# regime still matches.
CACHED_RETAIN_NPY = ROOT / "results/umap/embeddings/shared/retain.npy"
CACHED_RETAIN_TSV = ROOT / "results/umap/embeddings/shared/retain.tsv"
CACHED_RETAIN_MANIFEST = ROOT / "results/umap/embeddings/manifest.json"

N_REPS_EXPECTED = 35_459
N_RETAIN_EXPECTED = 194_064

CORPORA = ("reps", "retain")

# The study's own anchor families are in the reps corpus by construction. They are flagged
# everywhere rather than removed: at n=2 of 35,459 they cannot move an aggregate, but a
# reader must not discover their presence later.
ANCHORS = {"MGYF0000018690": "F", "MGYF0000027605": "F_prime"}

# 50-aa reporting bins; the final open bin absorbs everything past the truncation point.
LEN_EDGES: Tuple[int, ...] = tuple(range(50, 1050, 50)) + (10**9,)
MATCH_BIN_WIDTH = 10
MATCH_MAX_DELTA = 25

STD_AA = "ACDEFGHIKLMNPQRSTVWY"

# --------------------------------------------------------------------------- #
# Palette — Okabe-Ito, same hues as scripts/13_umap_plot.py so the figures of this
# experiment sit beside the existing UMAP figures without a colour clash.
# --------------------------------------------------------------------------- #
C_REPS = "#D55E00"        # vermillion
C_RETAIN = "#0072B2"      # blue
C_MATCHED = "#009E73"     # green
C_NOVEL = "#CC79A7"       # purple
SEQ_CMAP = "cividis"
DIV_CMAP = "PuOr"
HEXBIN = dict(gridsize=170, bins="log", cmap="Greys", mincnt=1, linewidths=0, alpha=0.55)

# UMAP recipe — identical to scripts/13_umap_plot.py so coordinates are comparable.
PCA_DIM = 50
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
KNN_K = 50


# --------------------------------------------------------------------------- #
# Corpus parsing
# --------------------------------------------------------------------------- #
@dataclass
class RepMeta:
    id: str
    length: int
    declared_len: Optional[int]
    rep: str
    full_size: Optional[int]
    source: str

    def as_row(self) -> dict:
        d = asdict(self)
        d["is_anchor"] = self.id in ANCHORS
        d["anchor_role"] = ANCHORS.get(self.id, "")
        d["truncated"] = self.length > SCORED_MAX
        return d


def parse_rep_header(header: str) -> Dict[str, str]:
    """Parse a reps FASTA description line (without the leading '>').

    Real format:
        MGYF0000000001 rep=MGYP006279162177/1-95 len=95 full_size=143098 source=tm_blob

    The id is the first whitespace token; every ``key=value`` token after it becomes a
    key. Never raises — a malformed or truncated header yields whatever it does carry, and
    t0 is the place that decides whether what is missing matters.
    """
    parts = str(header).strip().split()
    if not parts:
        return {}
    out: Dict[str, str] = {"id": parts[0]}
    for tok in parts[1:]:
        if "=" in tok:
            k, _, v = tok.partition("=")
            if k:
                out[k] = v
    return out


def _as_int(v: Optional[str]) -> Optional[int]:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_reps(path=REPS_FASTA, limit: Optional[int] = None):
    """Read the reps FASTA -> (records, metadata). Records keep FASTA order.

    ``src.utils.io.read_fasta`` keeps only the first header token as `id`, but the reps
    corpus carries ``len=``/``full_size=``/``source=`` after it, so the raw header lines
    are read in a second cheap pass and zipped back on.
    """
    from src.utils.io import read_fasta

    records = read_fasta(str(path))
    headers = read_fasta_headers(path)
    if len(headers) != len(records):
        raise SystemExit(f"{path}: {len(headers)} headers but {len(records)} records parsed")
    if limit:
        records, headers = records[:limit], headers[:limit]
    meta: List[RepMeta] = []
    for r, head in zip(records, headers):
        fields = parse_rep_header(head)
        meta.append(
            RepMeta(
                id=r.id,
                length=len(r.seq),
                declared_len=_as_int(fields.get("len")),
                rep=fields.get("rep", ""),
                full_size=_as_int(fields.get("full_size")),
                source=fields.get("source", ""),
            )
        )
    return records, meta


def read_fasta_headers(path) -> List[str]:
    """Full header lines (without '>') in file order.

    ``src.utils.io.read_fasta`` drops everything after the first whitespace token, but the
    reps corpus carries ``len=``/``full_size=``/``source=`` there, so this experiment needs
    the raw line. One cheap pass over the file.
    """
    heads: List[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                heads.append(line[1:].strip())
    return heads


# --------------------------------------------------------------------------- #
# Cost model and work partitioning
# --------------------------------------------------------------------------- #
def scored_positions(length: int, max_length: int = MAX_LENGTH) -> int:
    """Positions `pseudo_perplexity` will actually score after truncation."""
    return int(min(int(length), max_length - 2))


def predicted_cost(length: int, mpp: int = MASKS_PER_PASS, max_length: int = MAX_LENGTH) -> int:
    """Token-forwards needed to score one sequence: Lt * ceil(Lt / mpp).

    `_score_one` builds ceil(Lt/mpp) masked variants, each a full forward of length Lt, so
    cost grows ~L^2. This is what makes index-balanced sharding wrong and cost-balanced
    sharding necessary.
    """
    lt = scored_positions(length, max_length)
    if lt <= 0:
        return 0
    return int(lt * math.ceil(lt / max(1, int(mpp))))


def length_bin(length: int, edges: Sequence[int] = LEN_EDGES) -> int:
    """Index of the bin whose upper edge first exceeds ``length`` (bins are (lo, hi])."""
    for i, hi in enumerate(edges):
        if length <= hi:
            return i
    return len(edges) - 1


def shard_assign(lengths: Sequence[int], n_shards: int, mpp: int = MASKS_PER_PASS) -> np.ndarray:
    """Greedy least-loaded bin-packing of sequences onto shards by predicted cost.

    Deterministic: sequences are visited in descending cost (ties broken by index) and
    each goes to the currently cheapest shard. A contiguous or round-robin split would
    hand one shard the long tail and run it ~8x longer than the rest, which is how an
    array job times out at 23/24 tasks complete.
    """
    n_shards = int(n_shards)
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    costs = np.array([predicted_cost(int(L), mpp) for L in lengths], dtype=np.int64)
    order = np.lexsort((np.arange(costs.size), -costs))
    load = np.zeros(n_shards, dtype=np.int64)
    out = np.empty(costs.size, dtype=np.int32)
    for idx in order:
        s = int(np.argmin(load))
        out[idx] = s
        load[s] += costs[idx]
    return out


def shard_cost_table(lengths: Sequence[int], shards: np.ndarray, n_shards: int,
                     mpp: int = MASKS_PER_PASS) -> List[dict]:
    costs = np.array([predicted_cost(int(L), mpp) for L in lengths], dtype=np.int64)
    rows = []
    for s in range(int(n_shards)):
        m = shards == s
        rows.append({"shard": s, "n_seqs": int(m.sum()), "token_forwards": int(costs[m].sum())})
    return rows


# --------------------------------------------------------------------------- #
# Length matching
# --------------------------------------------------------------------------- #
def length_match(target_lengths: Sequence[int], pool_lengths: Sequence[int], *,
                 seed: int = SEED, bin_width: int = MATCH_BIN_WIDTH,
                 max_delta: int = MATCH_MAX_DELTA) -> Tuple[List[int], dict]:
    """1:1 without-replacement nearest-length match of ``pool`` rows to ``target`` rows.

    Pool indices are bucketed by ``L // bin_width`` into rng(seed)-shuffled stacks;
    targets are consumed LONGEST-FIRST, which is what makes the greedy pass succeed — the
    scarce long buckets get first claim. When a bucket is empty the search walks outward
    +-1, +-2 ... up to ``max_delta // bin_width`` buckets; past that the target is recorded
    as unmatched (-1) rather than matched badly.

    Returns (pool index per target, diagnostics).
    """
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target_lengths, dtype=np.int64)
    pool = np.asarray(pool_lengths, dtype=np.int64)
    bw = max(1, int(bin_width))

    buckets: Dict[int, List[int]] = {}
    for idx in rng.permutation(pool.size):
        buckets.setdefault(int(pool[idx]) // bw, []).append(int(idx))

    max_steps = max(0, int(max_delta) // bw)
    out = np.full(tgt.size, -1, dtype=np.int64)
    for t in np.argsort(-tgt, kind="stable"):
        home = int(tgt[t]) // bw
        picked = -1
        for step in range(max_steps + 1):
            for cand in ((home,) if step == 0 else (home - step, home + step)):
                stack = buckets.get(cand)
                if stack:
                    picked = stack.pop()
                    break
            if picked >= 0:
                break
        out[t] = picked

    matched = out >= 0
    d = np.abs(pool[out[matched]] - tgt[matched]) if matched.any() else np.array([0])
    diag = {
        "method": "greedy nearest-length, without replacement, longest-first",
        "bin_width": bw,
        "max_delta": int(max_delta),
        "seed": int(seed),
        "n_target": int(tgt.size),
        "n_pool": int(pool.size),
        "n_matched": int(matched.sum()),
        "n_unmatched": int((~matched).sum()),
        "mean_abs_dlen": float(d.mean()),
        "max_abs_dlen": int(d.max()),
    }
    return [int(x) for x in out], diag


# --------------------------------------------------------------------------- #
# Aggregation and statistics (numpy + scipy only; statsmodels is not installed)
# --------------------------------------------------------------------------- #
def micro_ppl(nll: Sequence[float], pos: Sequence[int]) -> float:
    """Corpus pseudo-perplexity exp(sum nll / sum positions).

    Identical to `src.eval.pseudo_perplexity.aggregate_nll(...).ppl` — the project's
    headline convention, asserted equal in test_mgreps.py.
    """
    total_nll = float(np.sum(np.asarray(nll, dtype=np.float64)))
    total_pos = int(np.sum(np.asarray(pos, dtype=np.int64)))
    if total_pos <= 0:
        return float("nan")
    return float(math.exp(total_nll / total_pos))


def bootstrap_ci(stat_fn, *arrays, n_boot: int = N_BOOT, seed: int = SEED,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of ``stat_fn(*arrays)`` resampling rows jointly."""
    arrs = [np.asarray(a) for a in arrays]
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs) or n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        vals[b] = float(stat_fn(*(a[idx] for a in arrs)))
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(finite, alpha / 2)), float(np.quantile(finite, 1 - alpha / 2)))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta = 2U/(n1*n2) - 1, i.e. P(a>b) - P(a<b). +1 fully above, 0 identical.

    Scale-free and robust — the effect size to read instead of the p-value, which at
    n=35,459 per arm is significant for any difference whatsoever.
    """
    from scipy.stats import mannwhitneyu

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    u = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").statistic
    return float(2.0 * u / (a.size * b.size) - 1.0)


def two_sample_stats(a: Sequence[float], b: Sequence[float], *,
                     a_nll=None, a_pos=None, b_nll=None, b_pos=None,
                     n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """One contrast block: log-PPL location shift, effect sizes, and the micro-PPL ratio.

    ``a`` and ``b`` are per-sequence log-PPL. When the nll/position arrays are supplied the
    micro-PPL ratio gets a bootstrap CI that re-aggregates exp(sum nll / sum pos) over the
    resampled sequences — precisely what PPLResult retains those fields for.
    """
    from scipy.stats import ks_2samp, mannwhitneyu

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mw = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    ks = ks_2samp(a, b)
    pooled_sd = math.sqrt(((a.var(ddof=1) * (a.size - 1)) + (b.var(ddof=1) * (b.size - 1)))
                          / max(1, a.size + b.size - 2)) if min(a.size, b.size) > 1 else float("nan")

    out = {
        "n_a": int(a.size), "n_b": int(b.size),
        "delta_log_ppl_mean": float(a.mean() - b.mean()),
        "delta_log_ppl_mean_ci95": list(bootstrap_ci(
            lambda x: float(np.mean(x)), a, n_boot=n_boot, seed=seed)),
        "delta_log_ppl_median": float(np.median(a) - np.median(b)),
        "cliffs_delta": float(2.0 * mw.statistic / (a.size * b.size) - 1.0),
        "cohens_d": float((a.mean() - b.mean()) / pooled_sd) if pooled_sd and math.isfinite(pooled_sd) else float("nan"),
        "mannwhitney_u": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "method": "percentile"},
    }
    # Difference-of-medians and -means CIs need both samples resampled independently.
    rng = np.random.default_rng(seed)
    dmed = np.empty(int(n_boot)); dmean = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        ia = rng.integers(0, a.size, size=a.size)
        ib = rng.integers(0, b.size, size=b.size)
        dmed[i] = np.median(a[ia]) - np.median(b[ib])
        dmean[i] = a[ia].mean() - b[ib].mean()
    out["delta_log_ppl_median_ci95"] = [float(np.quantile(dmed, 0.025)), float(np.quantile(dmed, 0.975))]
    out["delta_log_ppl_mean_ci95"] = [float(np.quantile(dmean, 0.025)), float(np.quantile(dmean, 0.975))]

    if a_nll is not None and b_nll is not None:
        an = np.asarray(a_nll, dtype=np.float64); ap = np.asarray(a_pos, dtype=np.float64)
        bn = np.asarray(b_nll, dtype=np.float64); bp = np.asarray(b_pos, dtype=np.float64)
        out["ppl_micro_a"] = micro_ppl(an, ap)
        out["ppl_micro_b"] = micro_ppl(bn, bp)
        out["ratio_micro_ppl"] = out["ppl_micro_a"] / out["ppl_micro_b"]
        ratios = np.empty(int(n_boot))
        rng = np.random.default_rng(seed + 1)
        for i in range(int(n_boot)):
            ia = rng.integers(0, an.size, size=an.size)
            ib = rng.integers(0, bn.size, size=bn.size)
            ratios[i] = micro_ppl(an[ia], ap[ia]) / micro_ppl(bn[ib], bp[ib])
        out["ratio_micro_ppl_ci95"] = [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]
    return out


def binned_contrast(len_a, log_a, len_b, log_b, edges: Sequence[int] = LEN_EDGES,
                    n_boot: int = 500, seed: int = SEED, min_n: int = 30) -> List[dict]:
    """Per-length-bin median difference with a bootstrap CI.

    The confound-proof readout: "reps are higher in k of N bins" does not depend on the
    corpora's marginal length distributions at all.
    """
    len_a = np.asarray(len_a); log_a = np.asarray(log_a, dtype=np.float64)
    len_b = np.asarray(len_b); log_b = np.asarray(log_b, dtype=np.float64)
    lo = 0
    rows: List[dict] = []
    for i, hi in enumerate(edges):
        ma = (len_a > lo) & (len_a <= hi)
        mb = (len_b > lo) & (len_b <= hi)
        row = {"bin": i, "lo": int(lo), "hi": int(hi) if hi < 10**8 else None,
               "n_reps": int(ma.sum()), "n_retain": int(mb.sum())}
        if ma.sum() >= min_n and mb.sum() >= min_n:
            aa, bb = log_a[ma], log_b[mb]
            row["median_log_ppl_reps"] = float(np.median(aa))
            row["median_log_ppl_retain"] = float(np.median(bb))
            row["delta"] = float(np.median(aa) - np.median(bb))
            rng = np.random.default_rng(seed + i)
            d = np.empty(int(n_boot))
            for j in range(int(n_boot)):
                d[j] = (np.median(aa[rng.integers(0, aa.size, aa.size)])
                        - np.median(bb[rng.integers(0, bb.size, bb.size)]))
            row["delta_ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        else:
            row.update({"median_log_ppl_reps": None, "median_log_ppl_retain": None,
                        "delta": None, "delta_ci95": None})
        rows.append(row)
        lo = hi
    return rows


def standardize_by_length(len_ref, len_other, log_other, edges: Sequence[int] = LEN_EDGES
                          ) -> Tuple[float, np.ndarray]:
    """Direct standardization: ``other``'s mean log-PPL reweighted to ``ref``'s length mix.

    Sampling-free, and independent of the greedy matching — two adjustments agreeing is
    much stronger evidence than either alone. Returns (adjusted mean, per-row weights).
    """
    len_ref = np.asarray(len_ref); len_other = np.asarray(len_other)
    log_other = np.asarray(log_other, dtype=np.float64)
    bins_ref = np.array([length_bin(int(L), edges) for L in len_ref])
    bins_oth = np.array([length_bin(int(L), edges) for L in len_other])
    nb = len(edges)
    ref_frac = np.bincount(bins_ref, minlength=nb).astype(np.float64)
    ref_frac /= max(1.0, ref_frac.sum())
    oth_count = np.bincount(bins_oth, minlength=nb).astype(np.float64)

    w = np.zeros(log_other.size, dtype=np.float64)
    for b in range(nb):
        if oth_count[b] > 0 and ref_frac[b] > 0:
            w[bins_oth == b] = ref_frac[b] / oth_count[b]
    if w.sum() <= 0:
        return float("nan"), w
    w = w / w.sum()
    return float(np.sum(w * log_other)), w


def ols_bootstrap(X: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """OLS via numpy lstsq with case-bootstrap coefficient CIs (statsmodels is absent)."""
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    rng = np.random.default_rng(seed)
    boots = np.empty((int(n_boot), X.shape[1]), dtype=np.float64)
    n = X.shape[0]
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[i], *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
    return {
        "coef": [float(v) for v in beta],
        "coef_ci95": [[float(np.quantile(boots[:, j], 0.025)),
                       float(np.quantile(boots[:, j], 0.975))] for j in range(X.shape[1])],
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(n),
        "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def aa_composition(seqs: Iterable[str]) -> Dict[str, float]:
    """Amino-acid frequencies over a corpus (all observed letters, not just the 20)."""
    counts: Dict[str, int] = {}
    total = 0
    for s in seqs:
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items())}


def seq_entropy(seq: str) -> float:
    """Shannon entropy (nats) of a sequence's own AA composition — low-complexity proxy."""
    if not seq:
        return float("nan")
    counts: Dict[str, int] = {}
    for ch in seq:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(seq)
    return float(-sum((c / n) * math.log(c / n) for c in counts.values()))


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence (nats) between two composition dicts. 0 iff identical."""
    keys = sorted(set(p) | set(q))
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    pv = pv / pv.sum() if pv.sum() > 0 else pv
    qv = qv / qv.sum() if qv.sum() > 0 else qv
    m = 0.5 * (pv + qv)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(pv, m) + 0.5 * _kl(qv, m)


# --------------------------------------------------------------------------- #
# Embedding-space separation
# --------------------------------------------------------------------------- #
def knn_mixing(X: np.ndarray, labels: np.ndarray, k: int = KNN_K, seed: int = SEED,
               metric: str = UMAP_METRIC) -> dict:
    """Fraction of each corpus's k nearest neighbours that come from the other corpus.

    This is the quantitative form of "same region or separate". The UMAP picture cannot
    answer it: inter-cluster distance in a UMAP embedding is not metric (the caveat
    scripts/13_umap_plot.py already carries). Computed in PCA-50 space under the same
    cosine metric the UMAP fit uses, and compared against the chance rate implied by the
    corpus sizes.
    """
    from pynndescent import NNDescent

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    labels = np.asarray(labels)
    index = NNDescent(X, metric=metric, n_neighbors=min(int(k) + 1, X.shape[0] - 1),
                      random_state=int(seed))
    index.prepare()
    nbrs, _ = index.neighbor_graph
    nbrs = nbrs[:, 1:]                       # drop self
    is_retain = (labels == "retain")
    nb_retain = is_retain[nbrs]
    frac_retain_nbrs = nb_retain.mean(axis=1)
    null = float(is_retain.mean())
    return {
        "knn_k": int(nbrs.shape[1]),
        "knn_metric": metric,
        "null_frac_retain": null,
        "mean_frac_retain_neighbours_of_reps": float(frac_retain_nbrs[~is_retain].mean()),
        "mean_frac_reps_neighbours_of_retain": float(1.0 - frac_retain_nbrs[is_retain].mean()),
        "null_frac_reps": float(1.0 - null),
        "note": "values near the null mean the corpora are interleaved; near 0/1 means separated",
    }


def centroid_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine between corpus centroids in the ORIGINAL embedding space.

    Deliberately not computed on UMAP coordinates — see knn_mixing's note.
    """
    ca = np.asarray(A, dtype=np.float64).mean(axis=0)
    cb = np.asarray(B, dtype=np.float64).mean(axis=0)
    d = np.linalg.norm(ca) * np.linalg.norm(cb)
    return float(ca @ cb / d) if d > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Model / embedding helpers (torch imported lazily)
# --------------------------------------------------------------------------- #
def device_arg(prefer: str = "cuda", allow_cpu_fallback: bool = False) -> str:
    """Resolve the device, and REFUSE to silently fall back to CPU.

    ``src.utils.io.get_device`` quietly returns "cpu" when CUDA is missing. That default
    is right for the main pipeline and dangerous here: a transient CUDA-init fault on an
    allocated node would turn a 20-minute shard into a multi-day CPU crawl that hits its
    wall having written nothing. Same rationale as
    ``experiments/5_readout_vs_representation/common.py::device_arg``.
    """
    from src.utils.io import get_device

    resolved = get_device(prefer)
    if prefer.startswith("cuda") and not resolved.startswith("cuda") and not allow_cpu_fallback:
        raise SystemExit(
            "--device cuda was requested but torch.cuda.is_available() is False.\n"
            f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '?')} "
            f"node={os.environ.get('SLURMD_NODENAME', '?')} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
            "  To run on CPU deliberately, pass --device cpu or --allow-cpu-fallback."
        )
    return resolved


def embed_length_sorted(model, tokenizer, seqs: List[str], device: str, batch_size: int,
                        max_length: int) -> np.ndarray:
    """``embed_sequences`` with length bucketing, then the permutation undone.

    Copied from scripts/12_embed_corpora.py (which produced the cached retain embeddings
    this experiment reuses) so both corpora go through a byte-identical pooling path. The
    scripts/ module name starts with a digit and cannot be imported; exp-5's common.py set
    the precedent of copying rather than importlib-ing.
    """
    import torch

    from src.eval.membership import embed_sequences

    order = np.argsort([len(s) for s in seqs], kind="stable")
    ordered = [seqs[i] for i in order]

    if device.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.float16):
            feats = embed_sequences(model, tokenizer, ordered, device=device,
                                    batch_size=batch_size, max_length=max_length)
    else:
        feats = embed_sequences(model, tokenizer, ordered, device=device,
                                batch_size=batch_size, max_length=max_length)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(feats)[inverse]


def hf_snapshot_dir(model_id: str = MODEL_ID) -> Optional[Path]:
    """The local HF cache snapshot for ``model_id``, or None — the offline-mode guard."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    base = home / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None
    snaps = [p for p in base.iterdir() if p.is_dir()]
    return snaps[0] if snaps else None


# --------------------------------------------------------------------------- #
# IO / CLI
# --------------------------------------------------------------------------- #
def results_dir(*parts: str) -> Path:
    d = RESULTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path) -> Path:
    from src.utils.io import save_json as _sj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _sj(_denan(obj), str(path))
    print(f"  wrote {path}", flush=True)
    return Path(path)


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


def _denan(o):
    if isinstance(o, dict):
        return {k: _denan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_denan(v) for v in o]
    if isinstance(o, np.generic):
        return _denan(o.item())
    if isinstance(o, np.ndarray):
        return [_denan(v) for v in o.tolist()]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def git_rev() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "n/a"
    except Exception:  # noqa: BLE001 — provenance is nice to have, never load-bearing
        return "n/a"


def protocol_block(**overrides) -> dict:
    """Self-describing provenance stamped into every artifact of this experiment."""
    block = {
        "experiment": "6_mgnify_reps_vs_retain",
        "model_id": MODEL_ID,
        "variant": "base (theta_0) — no task-blocked checkpoint is loaded in this experiment",
        "max_length": MAX_LENGTH,
        "scored_positions_max": SCORED_MAX,
        "eval_batch": EVAL_BATCH,
        "embed_batch": EMBED_BATCH,
        "masks_per_pass": MASKS_PER_PASS,
        "seed": SEED,
        "n_boot": N_BOOT,
        "reps_fasta": str(REPS_FASTA),
        "retain_fasta": str(RETAIN_FASTA),
        "cached_retain_npy": str(CACHED_RETAIN_NPY),
        "tag": TAG,
        "git_rev": git_rev(),
        "matches": "configs/config.yaml eval.{batch_size,masks_per_pass} + "
                   "scripts/12_embed_corpora.py embedding path",
    }
    block.update(overrides)
    return block


def common_args(parser):
    """Flags every phase script shares."""
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results", default=None,
                        help="override the results dir (default: results/ or results_<TAG>/)")
    return parser


def apply_results_override(path: Optional[str]) -> None:
    global RESULTS
    if path:
        RESULTS = Path(path)
    RESULTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def set_rc() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,          # editable text in the PDF
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 110, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.frameon": False,
    })


def style_axis(ax, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)


def save_fig(fig, stem: str) -> List[Path]:
    out = results_dir("plots")
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=150)
        paths.append(p)
        print(f"  wrote {p}", flush=True)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths
