"""FLOPs accounting, used to compute-match the control A' to the blocked model B.

We use the standard transformer training estimate

    C_forward_backward  ~=  6 * N * D

where N is the number of trainable parameters and D the number of processed
(non-pad) tokens; forward is ~2ND, backward ~4ND. This is an approximation. We
match A' to B on this estimate and log the *realized* tolerance (Phase 5), rather
than trusting it blindly. Wall-clock is recorded separately to report the
defender-side time cost (Phase 4).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field


def count_params(model, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def forward_flops(n_params: int, n_tokens: int) -> float:
    return 2.0 * n_params * n_tokens


def train_flops(n_params: int, n_tokens: int) -> float:
    """Forward + backward."""
    return 6.0 * n_params * n_tokens


@dataclass
class FlopCounter:
    """Accumulates an estimated FLOP budget and wall-clock for one training run."""

    n_params: int
    total_flops: float = 0.0
    total_tokens: int = 0
    optimizer_steps: int = 0
    _t0: float = field(default_factory=time.time, repr=False)

    def record(self, n_tokens: int, fwd_bwd: bool = True, is_optimizer_step: bool = False) -> None:
        """Record one model pass over ``n_tokens`` tokens."""
        self.total_flops += train_flops(self.n_params, n_tokens) if fwd_bwd else forward_flops(self.n_params, n_tokens)
        self.total_tokens += int(n_tokens)
        if is_optimizer_step:
            self.optimizer_steps += 1

    @property
    def wall_clock_s(self) -> float:
        return time.time() - self._t0

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("_t0", None)
        d["wall_clock_s"] = self.wall_clock_s
        return d


def steps_to_match(target_flops: float, per_step_flops: float) -> int:
    """How many optimizer steps reach ``target_flops`` at ``per_step_flops`` each."""
    if per_step_flops <= 0:
        raise ValueError("per_step_flops must be positive")
    return int(math.ceil(target_flops / per_step_flops))


def relative_gap(a: float, b: float) -> float:
    """|a - b| / b, the FLOP-match tolerance we log for A' vs B."""
    if b == 0:
        return float("inf") if a != 0 else 0.0
    return abs(a - b) / abs(b)
