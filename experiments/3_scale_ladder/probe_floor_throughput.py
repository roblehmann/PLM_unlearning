"""Throughput+memory probe for the floor fit at large scales, on the REAL anchor data.

The 650M/150M gate floors timed out because a 120-epoch full-FT at batch 16 is ~35
min/epoch (650M). The obvious lever is a bigger batch, but F_train has a long-sequence
tail (median 216, p90 751, max 1367 tokens), so at batch 16 the floor already peaks near
75 GiB — a blind batch bump OOMs. This measures peak allocation and per-step time for a
handful of (batch_size, max_length) configs on the actual F_train sequences, so the
relaunch batch is chosen from data, not guessed.

Prints, per config: peak GiB, sec/step, and the projected wall-clock for one 8192-seq
epoch and for a 120-epoch floor. OOM configs are reported, not fatal.

    python experiments/3_scale_ladder/probe_floor_throughput.py --scale 650M
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import common
from common import HERE, banner, get_scale


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "scale_ladder.yaml"))
    p.add_argument("--scale", default="650M")
    p.add_argument("--ladder-root", default=str(common.DEFAULT_LADDER_ROOT))
    p.add_argument("--steps", type=int, default=8, help="timed steps per config (after 2 warmup)")
    p.add_argument("--configs", nargs="+", default=["16x1024", "24x1024", "32x1024", "48x768", "64x512"],
                   help="batch x max_length pairs to try")
    p.add_argument("--amp", default=None, help="bfloat16|float16 to autocast the training step (matches the "
                   "gate's floor fit); default fp32")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from omegaconf import OmegaConf

    from src.data.datamodule import DataPaths, make_mlm_loader
    from src.models.load import load_mlm, load_tokenizer
    from src.train.mlm import mlm_loss
    from src.utils.io import fasta_seqs

    cfg = OmegaConf.load(args.config)
    scale = get_scale(args.scale)
    banner(f"FLOOR THROUGHPUT PROBE @ {args.scale}  ({scale.model_id})")
    if not torch.cuda.is_available():
        raise SystemExit("run through sbatch on a GPU node")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.amp) if args.amp else None
    print(f"  device VRAM  : {total_gb:.1f} GiB   torch {torch.__version__}   amp={args.amp or 'fp32'}")

    root = common.run_root(args.ladder_root, args.scale)
    paths = DataPaths(str(root / "data"))
    tok = load_tokenizer(scale.model_id)
    # Use the longest-tail sequences so the probe measures the worst-case padded batch, not
    # an average one — the floor OOMs on its heaviest batch, not its mean.
    all_seqs = sorted(fasta_seqs(paths.F_train), key=len, reverse=True)
    steps_per_epoch = lambda b: (8192 + b - 1) // b

    print(f"  {'config':>10} {'peak GiB':>9} {'sec/step':>9} {'min/epoch':>10} {'h/120ep':>9}   verdict")
    for spec in args.configs:
        b, L = (int(x) for x in spec.lower().split("x"))
        peak, sps = _time_config(torch, load_mlm, make_mlm_loader, mlm_loss, scale.model_id,
                                 all_seqs, tok, b, L, args.steps, amp)
        if peak is None:
            print(f"  {spec:>10} {'OOM':>9} {'-':>9} {'-':>10} {'-':>9}   does not fit")
            continue
        mins_ep = sps * steps_per_epoch(b) / 60.0
        h120 = mins_ep * 120 / 60.0
        fit = "OK" if peak < 0.9 * total_gb else "TIGHT (>90% VRAM)"
        print(f"  {spec:>10} {peak:9.1f} {sps:9.2f} {mins_ep:10.1f} {h120:9.1f}   {fit}")

    print("\n  Pick the fastest config with peak < ~90% VRAM. Note: shorter max_length truncates the "
          "long-seq tail, which changes the FLOOR (best-achievable) protocol — if you cap length for "
          "150M/650M, re-run 8M/35M the same way so the ladder stays protocol-consistent.")


def _time_config(torch, load_mlm, make_mlm_loader, mlm_loss, model_id, seqs, tok, batch, max_len, steps, amp=None):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = load_mlm(model_id, device="cuda").train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        loader = make_mlm_loader(seqs[: batch * (steps + 3)], tok, batch, 0.15, max_len, shuffle=False, seed=0)
        it = iter(loader)
        t0 = None
        done = 0
        for i in range(steps + 2):                       # 2 warmup, then timed
            try:
                batch_data = next(it)
            except StopIteration:
                it = iter(loader)
                batch_data = next(it)
            opt.zero_grad(set_to_none=True)
            if amp is not None:
                with torch.autocast(device_type="cuda", dtype=amp):
                    loss = mlm_loss(model, batch_data, "cuda")
            else:
                loss = mlm_loss(model, batch_data, "cuda")
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if i == 1:
                t0 = time.time()
            elif i > 1:
                done += 1
        dt = (time.time() - t0) / max(1, done)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    except torch.cuda.OutOfMemoryError:
        peak, dt = None, None
    finally:
        for v in ("model", "opt", "loader", "it"):
            if v in dir():
                pass
        gc.collect()
        torch.cuda.empty_cache()
    return peak, dt


if __name__ == "__main__":
    main()
