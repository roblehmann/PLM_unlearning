"""Resource probe for the 650M rung — does the heaviest step actually fit on this GPU?

Not an experiment: a pre-flight check so a 24-hour Phase C job does not die at hour six on
an out-of-memory error. Reports the card's real VRAM and the measured peak allocation for
the three shapes of work the ladder runs, at the *configured* batch sizes:

  1. **full-FT step**       — params + grads + AdamW states + activations (Phases B and E).
  2. **task-blocking step** — the tent-pole (Phase C): K inner adversary steps followed by
     an outer update, so two optimizer states are live at once.
  3. **pseudo-PPL eval**    — inference at ``metrics.eval_batch_size`` (every phase).

Exits non-zero if any of the three does not fit, printing the batch size that would.

    sbatch experiments/3_scale_ladder/probe_650m_fit.sbatch
    python experiments/3_scale_ladder/probe_650m_fit.py --scale 650M   # on a GPU node
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import common
from common import HERE, banner, get_scale


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "scale_ladder.yaml"))
    p.add_argument("--scale", default="650M")
    p.add_argument("--seq-len", type=int, default=320, help="typical anchor sequence length")
    p.add_argument("--train-batch", type=int, default=16, help="matches blocking.retain_batch / attacker batch_size")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from omegaconf import OmegaConf

    from src.data.datamodule import make_mlm_loader
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.models.load import load_mlm, load_tokenizer
    from src.train.mlm import mlm_loss
    from src.train.taskblock import train_taskblock

    cfg = OmegaConf.load(args.config)
    scale = get_scale(args.scale)
    m = cfg.metrics
    max_len, eval_bs = int(m.max_length), int(m.eval_batch_size)

    banner(f"RESOURCE PROBE @ {args.scale}  ({scale.model_id})")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible — run this through the sbatch wrapper on a GPU node")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024 ** 3
    print(f"  device       : {props.name}  ({total_gb:.1f} GiB VRAM, cc {props.major}.{props.minor})")
    print(f"  torch        : {torch.__version__}")
    print(f"  probing with : seq_len={args.seq_len}, train_batch={args.train_batch}, eval_batch={eval_bs}, "
          f"max_length={max_len}")

    seq = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAG"
           "EGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRY")
    seqs = [(seq * ((args.seq_len // len(seq)) + 1))[: args.seq_len] for _ in range(args.train_batch * 2)]
    tok = load_tokenizer(scale.model_id)
    results = {}

    # ---- 1. full fine-tuning step ------------------------------------------------- #
    def full_ft_step():
        model = load_mlm(scale.model_id, device="cuda")
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        loader = make_mlm_loader(seqs, tok, args.train_batch, 0.15, max_len, shuffle=False, seed=0)
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            mlm_loss(model, batch, "cuda").backward()
            opt.step()
            break
        del model, opt, loader

    results["full_ft_step"] = _peak(torch, full_ft_step)

    # ---- 2. task-blocking outer step (the tent-pole) ------------------------------- #
    def taskblock_step():
        model = load_mlm(scale.model_id, device="cuda")
        block = OmegaConf.create(OmegaConf.to_container(cfg.task_blocking, resolve=True))
        train_taskblock(
            model, tok, seqs, seqs,
            OmegaConf.create({"inner_steps_K": int(block.inner_steps_K), "eta_inner": float(block.inner_lr),
                              "eta_outer": float(block.outer_lr), "lambda_tr": 1.0,
                              "forget_margin": float(block.forget_margin_nats), "outer_steps": 2,
                              "retain_batch": args.train_batch, "forget_batch": args.train_batch,
                              "warmup_steps": 1, "mlm_probability": 0.15, "grad_clip": 1.0, "log_every": 1}),
            "cuda", seed=0, max_length=max_len)
        del model

    results["taskblock_outer_step"] = _peak(torch, taskblock_step)

    # ---- 3. pseudo-perplexity evaluation ------------------------------------------- #
    def eval_step():
        model = load_mlm(scale.model_id, device="cuda")
        pseudo_perplexity(model, tok, seqs[:4], "cuda", eval_bs, max_len,
                          masks_per_pass=int(m.masks_per_pass))
        del model

    results["pseudo_ppl_eval"] = _peak(torch, eval_step)

    banner("PEAK GPU MEMORY")
    ok = True
    for name, peak in results.items():
        if peak is None:
            print(f"  {name:<24} OOM / failed")
            ok = False
            continue
        print(f"  {name:<24} {peak:6.1f} GiB peak   ({100 * peak / total_gb:.0f}% of {total_gb:.0f} GiB)")
        if peak > 0.9 * total_gb:
            print(f"    ! within 10% of capacity — halve the batch size for this stage")
            ok = False
    headroom = total_gb - max(p for p in results.values() if p is not None) if any(results.values()) else 0
    print(f"\n  spare VRAM at the heaviest stage: {headroom:.1f} GiB")
    print("  Phase C (task blocking) is the tent-pole; if it is the one that is tight, lower "
          "blocking.retain_batch / forget_batch rather than the eval batch.")
    if not ok:
        raise SystemExit(1)
    print(f"\n  => {args.scale} fits on this card at the configured batch sizes.")


def _peak(torch, fn) -> float | None:
    """Run ``fn`` and return its peak CUDA allocation in GiB (None if it OOMs)."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM in {getattr(fn, '__name__', fn)}")
        return None
    except Exception:
        traceback.print_exc()
        return None
    finally:
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        gc.collect()
        torch.cuda.empty_cache()
    return peak


if __name__ == "__main__":
    main()
