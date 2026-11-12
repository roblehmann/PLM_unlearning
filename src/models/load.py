"""Load HF ESM-2 as a masked LM, with optional LoRA attachment for the attacker.

All heavy imports (torch/transformers/peft) are lazy so the module imports in a
minimal environment (e.g. for test collection or `--help`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class LoraSpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None  # default below

    @classmethod
    def from_cfg(cls, cfg) -> "LoraSpec":
        return cls(
            rank=int(cfg.lora_rank),
            alpha=int(cfg.lora_alpha),
            dropout=float(cfg.lora_dropout),
            target_modules=list(cfg.target_modules),
        )


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def load_mlm(model_name: str, device: str = "cpu"):
    """Load an `EsmForMaskedLM` and move it to ``device``."""
    import torch
    from transformers import EsmForMaskedLM

    model = EsmForMaskedLM.from_pretrained(model_name)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return model.to(device)


def attach_lora(model, spec: LoraSpec):
    """Wrap ``model`` with LoRA adapters on ``spec.target_modules`` and freeze the base.

    target_modules are matched by suffix, so 'dense' covers the attention output
    and the FFN dense layers as well — a deliberately broad adapter set.
    """
    from peft import LoraConfig, get_peft_model

    targets = spec.target_modules or ["query", "key", "value", "dense"]
    lcfg = LoraConfig(
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=targets,
        bias="none",
        task_type=None,  # generic HF model with a custom MLM head; we call model(**batch).loss
    )
    return get_peft_model(model, lcfg)


def build_model(model_cfg, device: str = "cpu", lora: Optional[LoraSpec] = None) -> Tuple[Any, Any]:
    """Return (model, tokenizer). If ``lora`` is given, attach adapters (attacker)."""
    tokenizer = load_tokenizer(model_cfg.name)
    model = load_mlm(model_cfg.name, device=device)
    if lora is not None:
        model = attach_lora(model, lora)
    return model, tokenizer


def load_checkpoint(path: str, device: str = "cpu"):
    """Load a previously saved full model (A / B / A')."""
    import torch
    from transformers import EsmForMaskedLM

    model = EsmForMaskedLM.from_pretrained(path)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return model.to(device)


def save_model(model, path: str) -> None:
    """Save a full model or a LoRA adapter (PeftModel.save_pretrained handles both)."""
    model.save_pretrained(path)


def num_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
