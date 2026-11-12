"""Benign utility: MLM pseudo-perplexity on the retain (R) holdout — the headline
benign metric. Plus the B-vs-A' retention check and an optional downstream probe
(off by default; the minimal run does not need it)."""
from __future__ import annotations

from typing import Sequence

from src.eval.pseudo_perplexity import PPLResult, pseudo_perplexity


def benign_pseudo_perplexity(
    model,
    tokenizer,
    retain_holdout: Sequence[str],
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 1024,
    masks_per_pass: int = 1,
) -> PPLResult:
    return pseudo_perplexity(
        model, tokenizer, retain_holdout, device, batch_size, max_length, masks_per_pass
    )


def utility_retention(ppl_B: float, ppl_Aprime: float) -> float:
    """|PPL_benign(B) - PPL_benign(A')| / PPL_benign(A')."""
    if ppl_Aprime == 0:
        return float("inf")
    return abs(ppl_B - ppl_Aprime) / abs(ppl_Aprime)


def within_tolerance(ppl_B: float, ppl_Aprime: float, tolerance: float) -> bool:
    return utility_retention(ppl_B, ppl_Aprime) <= tolerance


def downstream_probe(*args, **kwargs):  # pragma: no cover - intentionally off in the minimal run
    """Optional secondary-structure / contact-prediction probe.

    Kept behind a flag and unimplemented in the minimal harness; the benign PPL
    above is the headline utility metric. Implement here if a downstream readout is
    needed, mirroring the membership probe (frozen embeddings + a small head)."""
    raise NotImplementedError(
        "Downstream utility probe is optional and disabled in the minimal run. "
        "Enable + implement only if benign PPL is insufficient."
    )
