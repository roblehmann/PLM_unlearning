"""Pseudo-perplexity for ESM-2 (a masked LM).

For a sequence x of length L, the pseudo-log-likelihood (Salazar et al. 2020) is

    PLL(x) = sum_i  log p(x_i | x_{\\i})

obtained by masking each residue position in turn. Pseudo-perplexity is

    PPL(x) = exp( -PLL(x) / L ).

Exact scoring costs L forward passes per sequence. We expose ``masks_per_pass`` to
mask several *strided* positions per forward pass (an approximation that trades a
small amount of accuracy for ~masks_per_pass speedup); ``masks_per_pass=1`` is exact.

Aggregation across a set of sequences is reported two ways:
  * ``ppl`` (micro / corpus): exp(sum NLL / sum positions)   <- headline
  * ``ppl_macro``: mean over sequences of per-sequence PPL.
The harness uses ``ppl`` everywhere (threshold τ, multiplier) for consistency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Pure aggregation / threshold math (no torch; unit-tested directly).
# --------------------------------------------------------------------------- #
@dataclass
class PPLResult:
    ppl: float                       # micro / corpus-level pseudo-perplexity
    ppl_macro: float                 # mean of per-sequence pseudo-perplexity
    per_seq_ppl: List[float] = field(default_factory=list)
    # Retained so a caller can re-aggregate a *corpus-level* PPL over a resampled set of
    # sequences (the bootstrap SE the scale-ladder headroom gate needs); the headline
    # `ppl` is exp(sum nll / sum positions), which is not a mean of `per_seq_ppl`.
    per_seq_nll: List[float] = field(default_factory=list)
    per_seq_positions: List[int] = field(default_factory=list)
    n_seqs: int = 0
    n_positions: int = 0
    masks_per_pass: int = 1

    def as_dict(self) -> dict:
        return {
            "ppl": self.ppl,
            "ppl_macro": self.ppl_macro,
            "n_seqs": self.n_seqs,
            "n_positions": self.n_positions,
            "masks_per_pass": self.masks_per_pass,
        }


def aggregate_nll(per_seq_nll: Sequence[float], per_seq_positions: Sequence[int], masks_per_pass: int = 1) -> PPLResult:
    """Combine per-sequence summed NLL + position counts into a PPLResult."""
    if len(per_seq_nll) != len(per_seq_positions):
        raise ValueError("per_seq_nll and per_seq_positions must align")
    total_nll = float(sum(per_seq_nll))
    total_pos = int(sum(per_seq_positions))
    per_seq_ppl = [
        math.exp(nll / pos) if pos > 0 else float("nan")
        for nll, pos in zip(per_seq_nll, per_seq_positions)
    ]
    ppl = math.exp(total_nll / total_pos) if total_pos > 0 else float("nan")
    finite = [p for p in per_seq_ppl if math.isfinite(p)]
    ppl_macro = float(sum(finite) / len(finite)) if finite else float("nan")
    return PPLResult(
        ppl=ppl,
        ppl_macro=ppl_macro,
        per_seq_ppl=per_seq_ppl,
        per_seq_nll=[float(x) for x in per_seq_nll],
        per_seq_positions=[int(x) for x in per_seq_positions],
        n_seqs=len(per_seq_nll),
        n_positions=total_pos,
        masks_per_pass=masks_per_pass,
    )


def acquisition_threshold(ppl_naive: float, ppl_floor: float, tau_rho: float) -> float:
    """τ = PPL_naive − tau_rho · (PPL_naive − PPL_floor).

    PPL_naive: base model A, no fine-tuning (worst / highest PPL on F).
    PPL_floor: A fine-tuned on all of F_train to convergence (best achievable).
    A model has "acquired" F at budget n iff its F_test PPL ≤ τ.
    """
    if not (0.0 <= tau_rho <= 1.0):
        raise ValueError("tau_rho must be in [0, 1]")
    return ppl_naive - tau_rho * (ppl_naive - ppl_floor)


def is_acquired(ppl: float, tau: float) -> bool:
    return ppl <= tau


# --------------------------------------------------------------------------- #
# Model-dependent scoring (torch; imported lazily).
# --------------------------------------------------------------------------- #
def _strided_groups(positions: List[int], masks_per_pass: int) -> List[List[int]]:
    """Partition positions into ``ceil(len/masks_per_pass)`` strided groups, so masked
    positions within a pass are spread out (reduces masked-token interaction bias)."""
    if masks_per_pass <= 1:
        return [[p] for p in positions]
    n_groups = math.ceil(len(positions) / masks_per_pass)
    groups: List[List[int]] = [[] for _ in range(n_groups)]
    for idx, p in enumerate(positions):
        groups[idx % n_groups].append(p)
    return [g for g in groups if g]


def _score_one(model, input_ids, special_mask, mask_token_id, device, batch_size, masks_per_pass):
    """Return (summed_nll, n_positions) for a single tokenized sequence."""
    import torch
    import torch.nn.functional as F

    score_positions = [i for i, m in enumerate(special_mask) if m == 0]
    if not score_positions:
        return 0.0, 0
    groups = _strided_groups(score_positions, masks_per_pass)
    base = torch.tensor(input_ids, dtype=torch.long)
    true_ids = base.clone()

    # Each group -> one masked variant; all variants share length, so we can stack.
    variants = []
    masked_positions = []
    for g in groups:
        v = base.clone()
        for p in g:
            v[p] = mask_token_id
        variants.append(v)
        masked_positions.append(g)

    # Accumulate NLL on-device (one .item() per sequence, not per residue — avoids a
    # GPU<->CPU sync per position, which dominates on large eval sets).
    total = None
    for start in range(0, len(variants), batch_size):
        chunk = variants[start : start + batch_size]
        chunk_pos = masked_positions[start : start + batch_size]
        batch = torch.stack(chunk).to(device)
        attn = torch.ones_like(batch)
        logits = model(input_ids=batch, attention_mask=attn).logits  # [b, L, V]
        logp = F.log_softmax(logits.float(), dim=-1)
        true = true_ids.to(device).unsqueeze(0).expand(batch.size(0), -1)  # [b, L]
        tok_logp = logp.gather(-1, true.unsqueeze(-1)).squeeze(-1)          # [b, L]
        score_mask = torch.zeros_like(batch, dtype=torch.bool)
        for row, positions in enumerate(chunk_pos):
            score_mask[row, positions] = True
        chunk_nll = -(tok_logp[score_mask]).sum()
        total = chunk_nll if total is None else total + chunk_nll
    return (float(total.item()) if total is not None else 0.0), len(score_positions)


def pseudo_perplexity(
    model,
    tokenizer,
    sequences: Sequence[str],
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 1024,
    masks_per_pass: int = 1,
) -> PPLResult:
    """Compute pseudo-perplexity over ``sequences``. Runs under no_grad in eval mode."""
    import torch

    was_training = model.training
    model.eval()
    mask_id = tokenizer.mask_token_id
    per_seq_nll: List[float] = []
    per_seq_pos: List[int] = []
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(
                seq,
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )
            nll, n = _score_one(
                model,
                enc["input_ids"],
                enc["special_tokens_mask"],
                mask_id,
                device,
                batch_size,
                masks_per_pass,
            )
            per_seq_nll.append(nll)
            per_seq_pos.append(n)
    if was_training:
        model.train()
    res = aggregate_nll(per_seq_nll, per_seq_pos, masks_per_pass=masks_per_pass)
    return res
