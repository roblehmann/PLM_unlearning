"""Plain masked-LM continued pretraining.

Two roles:
  * produce the compute-matched control A' (Phase 5) by continuing MLM on the
    retain set R for a target FLOP budget;
  * provide the shared MLM loss / token-count helpers used by the attacker and the
    task-blocking trainer.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from src.data.datamodule import cycle, make_mlm_loader
from src.utils.flops import FlopCounter
from src.utils.seeding import set_seed


def cfg_get(cfg, key: str, default):
    """Read a key from an OmegaConf node or a plain dict, with a default."""
    if cfg is None:
        return default
    if hasattr(cfg, "get") and not isinstance(cfg, (list, tuple)):
        try:
            v = cfg.get(key, default)
            return default if v is None else v
        except Exception:  # noqa: BLE001
            pass
    return getattr(cfg, key, default)


def mlm_loss(model, batch, device: str):
    """MLM loss for one collated batch (input_ids, attention_mask, labels)."""
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return model(**inputs).loss


def batch_tokens(batch) -> int:
    """Number of non-pad tokens in a batch (for FLOP accounting)."""
    return int(batch["attention_mask"].sum().item())


def _warmup_scheduler(optimizer, warmup_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / float(warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def continued_pretrain(
    model,
    tokenizer,
    sequences: Sequence[str],
    device: str,
    *,
    lr: float,
    batch_size: int,
    mlm_probability: float = 0.15,
    max_length: int = 1024,
    max_steps: Optional[int] = None,
    target_flops: Optional[float] = None,
    seed: int = 0,
    grad_clip: float = 1.0,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    tracker=None,
    log_every: int = 25,
) -> dict:
    """Continue MLM training. Stop at ``max_steps`` and/or when the estimated FLOP
    budget reaches ``target_flops`` (at least one must be set). Returns the trained
    model plus a FLOP/step/wall-clock record (used to match A' to B)."""
    import torch

    if max_steps is None and target_flops is None:
        raise ValueError("Set max_steps and/or target_flops")
    set_seed(seed)
    model.train().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = _warmup_scheduler(optimizer, warmup_steps)
    counter = FlopCounter(n_params=sum(p.numel() for p in params))
    loader = make_mlm_loader(sequences, tokenizer, batch_size, mlm_probability, max_length, shuffle=True, seed=seed)

    history: List[dict] = []
    step = 0
    for batch in cycle(loader):
        optimizer.zero_grad(set_to_none=True)
        loss = mlm_loss(model, batch, device)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        scheduler.step()
        counter.record(batch_tokens(batch), fwd_bwd=True, is_optimizer_step=True)
        step += 1
        if step % log_every == 0:
            rec = {"step": step, "mlm_loss": float(loss.item()), "flops": counter.total_flops}
            history.append(rec)
            if tracker:
                tracker.log(rec, step=step)
        if max_steps is not None and step >= max_steps:
            break
        if target_flops is not None and counter.total_flops >= target_flops:
            break

    return {"model": model, "steps": step, "flops": counter.as_dict(), "history": history}
