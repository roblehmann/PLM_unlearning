"""Shared plumbing for the 650M scale spot-check ("scaling ladder of the mechanism").

The claim this experiment defends is *not* "M is large at 650M". It is: the in-basin
geometry that **explains** M≈1 at 35M (sub-control displacement + a near-zero
linear-mode-connectivity barrier) is scale-robust across 8M -> 35M -> 150M -> 650M,
anchored by one real M measurement at 650M.

Design invariants enforced here:

* **Scale is a parameter, never a fork.** Every phase runs the *same* code at every
  scale; only :data:`SCALES` and the Hydra ``model=`` group differ. Nothing branches on
  layer count — 6/12/30/33-layer models are reconciled at *reporting* time by mapping
  each layer to relative depth in [0, 1] (:func:`relative_depth`).
* **Per-scale run roots.** Each scale gets its own ``paths.root`` under the ladder root
  (``<ladder_root>/<scale>``), symlinking one shared data root, so the existing pipeline
  scripts (``scripts/0{2,3,4,5,6,8}_*.py``) run unmodified with
  ``model=esm2_650m paths.root=runs/scale_ladder/650M``.
* **Manifests are the only source of truth.** Phases append rows to
  ``<ladder_root>/manifests/*.json``; the figure and table are regenerated from those
  alone, so no number reaches the paper by hand.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # repo root (taskblock-poc/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TASKBLOCK_ROOT", str(ROOT))

# Default ladder root and the anchor data root. The anchor is the MGnify metagenomic
# family — the only F in this repo that clears the headroom gate at 35M on properly
# decontaminated data (rel_headroom 0.29); the Rossmann Pfam anchors all failed it
# because base ESM-2 already models them cold.
DEFAULT_LADDER_ROOT = ROOT / "runs" / "scale_ladder"
DEFAULT_DATA_ROOT = ROOT / "runs" / "MGYF18690_metagenomic" / "data"

# ESM-2 random-prediction perplexity ceiling: 20 residue types, uniform prediction
# (ESM-2 paper's definition).
RANDOM_CEILING = 20.0


class Scale(dict):
    """One rung of the ladder. dict-backed so it serialises straight into a manifest."""

    def __getattr__(self, k):                      # scale.model_id reads better inline
        try:
            return self[k]
        except KeyError as e:                      # pragma: no cover - programming error
            raise AttributeError(k) from e


# ``val_ppl_reference`` = the UR50 validation perplexity reported in ESM-2 Table 1. It is
# a *reference* (not measured here) that fixes each rung's absolute language-model
# quality, so a reader can see the ladder spans a real capability range.
SCALES: Dict[str, Scale] = {
    "8M":   Scale(scale="8M",   model_id="facebook/esm2_t6_8M_UR50D",     hydra_model="esm2_8m",
                  num_layers=6,  embed_dim=320,  n_params=8_000_000,   val_ppl_reference=10.33),
    "35M":  Scale(scale="35M",  model_id="facebook/esm2_t12_35M_UR50D",   hydra_model="esm2_35m",
                  num_layers=12, embed_dim=480,  n_params=35_000_000,  val_ppl_reference=8.95),
    "150M": Scale(scale="150M", model_id="facebook/esm2_t30_150M_UR50D",  hydra_model="esm2_150m",
                  num_layers=30, embed_dim=640,  n_params=150_000_000, val_ppl_reference=7.75),
    "650M": Scale(scale="650M", model_id="facebook/esm2_t33_650M_UR50D",  hydra_model="esm2_650m",
                  num_layers=33, embed_dim=1280, n_params=650_000_000, val_ppl_reference=6.95),
}

LADDER_ORDER = ["8M", "35M", "150M", "650M"]


def get_scale(name: str) -> Scale:
    if name not in SCALES:
        raise SystemExit(f"unknown scale {name!r}; known: {', '.join(LADDER_ORDER)}")
    return SCALES[name]


def sort_scales(names: Iterable[str]) -> List[str]:
    """Ladder order (by parameter count), not lexicographic ('150M' < '35M' as text)."""
    return sorted(set(names), key=lambda s: SCALES[s]["n_params"] if s in SCALES else math.inf)


# --------------------------------------------------------------------------- #
# Per-scale run roots
# --------------------------------------------------------------------------- #
def run_root(ladder_root: os.PathLike | str, scale: str) -> Path:
    return Path(ladder_root) / scale


def manifest_dir(ladder_root: os.PathLike | str) -> Path:
    d = Path(ladder_root) / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_run_root(ladder_root: os.PathLike | str, scale: str, data_root: os.PathLike | str) -> Path:
    """Create ``<ladder_root>/<scale>/{data -> data_root, checkpoints, results}``.

    ``data`` is a symlink, so every rung trains and is scored on the *identical* splits —
    the ladder varies model scale and nothing else. (Anchor re-selection at a scale, after
    a gate failure, is expressed by pointing that scale's root at a
    different data root — hence a per-scale link rather than one shared parent.)
    """
    root = run_root(ladder_root, scale)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    link, target = root / "data", Path(data_root).resolve()
    if not target.exists():
        raise SystemExit(f"data root does not exist: {target}")
    if link.is_symlink():
        if link.resolve() != target:
            link.unlink()
            link.symlink_to(target)
    elif link.exists():
        raise SystemExit(f"{link} exists and is not a symlink; refusing to clobber it")
    else:
        link.symlink_to(target)
    return root


# --------------------------------------------------------------------------- #
# Manifest IO — idempotent, order-stable, byte-reproducible
# --------------------------------------------------------------------------- #
def load_manifest(path: os.PathLike | str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"rows": [], "meta": {}}
    with open(p) as fh:
        return json.load(fh)


def save_manifest(path: os.PathLike | str, manifest: dict) -> Path:
    """Write with sorted keys + ladder-ordered rows so re-runs diff byte-for-byte."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = manifest.get("rows", [])
    manifest["rows"] = sorted(rows, key=lambda r: SCALES.get(r.get("scale", ""), {}).get("n_params", math.inf))
    with open(p, "w") as fh:
        json.dump(_denan(manifest), fh, indent=2, sort_keys=True, default=_jsonable)
        fh.write("\n")
    return p


def _denan(o):
    """NaN/Inf -> null. Python's json emits bare ``NaN``, which is not valid JSON and
    chokes any non-Python consumer of these manifests (jq, the LaTeX build, a reviewer)."""
    if isinstance(o, dict):
        return {k: _denan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_denan(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def upsert_row(path: os.PathLike | str, row: dict, key: str = "scale", meta: Optional[dict] = None) -> Path:
    """Replace the row with the same ``key`` (re-running one scale must not duplicate it).

    The read-modify-write is guarded by an exclusive lock on a sidecar ``.lock`` file, so
    the per-scale SLURM array tasks that write this manifest concurrently cannot lose each
    other's rows. Without it, two tasks finishing together would each write the manifest
    they read *before* the other's row landed, and one scale would silently vanish.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = p.with_suffix(p.suffix + ".lock")
    with open(lock, "w") as lf:
        _flock(lf)
        man = load_manifest(p)
        man["rows"] = [r for r in man.get("rows", []) if r.get(key) != row.get(key)] + [row]
        if meta:
            man.setdefault("meta", {}).update(meta)
        return save_manifest(p, man)


def _flock(fh) -> None:
    """Exclusive advisory lock, released when the handle closes. No-op where fcntl is
    unavailable (Windows) — the ladder only ever runs multi-writer under SLURM on Linux."""
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except (ImportError, OSError):
        pass


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def row_for(path: os.PathLike | str, scale: str) -> Optional[dict]:
    for r in load_manifest(path).get("rows", []):
        if r.get("scale") == scale:
            return r
    return None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def corpus_ppl(per_seq_nll: Sequence[float], per_seq_positions: Sequence[int]) -> float:
    """Aggregation: exp(total NLL / total scored positions)."""
    tot_n, tot_p = float(sum(per_seq_nll)), int(sum(per_seq_positions))
    return math.exp(tot_n / tot_p) if tot_p > 0 else float("nan")


def bootstrap_ppl_se(
    per_seq_nll: Sequence[float], per_seq_positions: Sequence[int], n_boot: int = 400, seed: int = 42
) -> float:
    """SE of the corpus-level pseudo-perplexity, by resampling *sequences* with
    replacement and re-aggregating each resample at corpus level.

    This is the noise scale the gate's ``tau_floor_margin_ses`` is measured in: if τ sits
    fewer than ~3 SEs above the floor, "crossed τ" is not distinguishable from "sampling
    noise on the floor" and no multiplier computed against it is trustworthy.
    """
    n = len(per_seq_nll)
    if n < 2:
        return float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(corpus_ppl([per_seq_nll[i] for i in idx], [per_seq_positions[i] for i in idx]))
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def plateaued(history: Sequence[dict], key: str = "target_ppl", rel_tol: float = 0.01, window: int = 3) -> bool:
    """Convergence by *acquisition-curve plateau*, not by hitting the max budget.

    True iff the last ``window`` evaluations improved the running best by less than
    ``rel_tol`` in relative terms. This is a hard gate input: a floor that is
    still falling when the budget runs out means τ is defined against an over-estimate of
    the achievable PPL, so every crossing budget measured against it is biased.
    """
    vals = [float(h[key]) for h in history if key in h and math.isfinite(float(h[key]))]
    if len(vals) < window + 1:
        return False
    best_before = min(vals[: -window])
    best_after = min(vals)
    return (best_before - best_after) / best_before < rel_tol if best_before > 0 else False


def relative_depth(layer_index: int, num_layers: int) -> float:
    """Map an encoder layer to depth in [0, 1] so 6/12/30/33-layer profiles overlay."""
    return 0.0 if num_layers <= 1 else layer_index / (num_layers - 1)


def log_grid(lo: int, hi: int, n_points: int) -> List[int]:
    """``n_points`` log-spaced integer budgets in [lo, hi], deduped and ascending."""
    if n_points < 2:
        return [int(hi)]
    step = (math.log(hi) - math.log(lo)) / (n_points - 1)
    out = sorted({int(round(math.exp(math.log(lo) + i * step))) for i in range(n_points)})
    return out


def banner(title: str) -> None:
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74, flush=True)
