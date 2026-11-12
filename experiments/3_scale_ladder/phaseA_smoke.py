"""Phase A of the 650M spot-check — SCALE PARAMETERIZATION SMOKE TEST.

Proves the pipeline has no hard-coded shapes before any real compute is spent: the same
code path is exercised end to end at 8M -> 35M -> 150M -> 650M on a handful of sequences
with toy budgets. What it touches, in order, is exactly what the real phases touch —
tokenizer, masked-LM forward, pseudo-perplexity, attacker fine-tune, task-blocking inner
adversary + outer update, FLOP-matched control, state-dict displacement, LMC
interpolation — so any assumption of 12 layers / width 480 / 35M checkpoint layout blows
up here, cheaply, instead of six GPU-hours into a 650M safeguard run.

    python experiments/3_scale_ladder/phaseA_smoke.py --scales 8M 35M 150M 650M

Everything is written under ``<ladder_root>/_smoke/<scale>`` and is disposable; the
throwaway manifest row per scale exists only to prove the manifest plumbing round-trips.
"""
from __future__ import annotations

import argparse
import shutil
import traceback
from pathlib import Path

import common
from common import DEFAULT_LADDER_ROOT, HERE, banner, get_scale, sort_scales


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "scale_ladder.yaml"))
    p.add_argument("--scales", nargs="+", default=None)
    p.add_argument("--ladder-root", default=str(DEFAULT_LADDER_ROOT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-seqs", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--keep", action="store_true", help="keep the throwaway <ladder_root>/_smoke tree")
    return p.parse_args()


TOY = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
       "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")


def main():
    args = parse_args()
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    scales = sort_scales(args.scales or list(cfg.scales))
    smoke_root = Path(args.ladder_root) / "_smoke"
    manifest = smoke_root / "smoke_manifest.json"
    if manifest.exists():
        manifest.unlink()

    failures = []
    for name in scales:
        banner(f"PHASE A — SMOKE @ {name}  ({get_scale(name).model_id})")
        try:
            row = smoke_one(get_scale(name), args, cfg, smoke_root)
            common.upsert_row(manifest, row, meta={"phase": "A", "disposable": True})
            print(f"  OK: {row['n_params_actual']:,} params, {row['num_layers_actual']} layers, "
                  f"width {row['embed_dim_actual']}")
        except Exception as e:                       # keep going: report every broken rung
            failures.append((name, repr(e)))
            traceback.print_exc()
            print(f"  FAILED @ {name}: {e!r}")

    banner("PHASE A SUMMARY")
    for r in common.load_manifest(manifest).get("rows", []):
        print(f"  {r['scale']:>5}  layers={r['num_layers_actual']:>2} width={r['embed_dim_actual']:>4}  "
              f"ppl={r['toy_ppl']:.2f}  disp={r['toy_disp_rel']:.2e}  lmc_ok={r['toy_lmc_ok']}")
    if not args.keep and smoke_root.exists():
        shutil.rmtree(smoke_root, ignore_errors=True)
        print(f"  (removed {smoke_root}; --keep to retain)")
    if failures:
        print("\n  FAILURES:")
        for n, e in failures:
            print(f"    {n}: {e}")
        raise SystemExit(1)
    print("\n  All rungs run the same code path with no shape/precision errors. "
          "Cleared to run Phase B (headroom gate).")


def smoke_one(scale, args, cfg, smoke_root: Path) -> dict:
    """One full pipeline pass at toy scale. Deliberately touches every stage that could
    hide a shape assumption, in the order the real phases hit them."""
    import torch
    from omegaconf import OmegaConf

    from src.analysis.weightspace import l2_by_group, lmc_path, model_state_dict_cpu
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.models.load import load_mlm, load_tokenizer, save_model
    from src.train.attack import run_attack
    from src.train.mlm import continued_pretrain
    from src.train.taskblock import train_taskblock
    from src.utils.seeding import set_seed

    seed = int(cfg.seed)
    set_seed(seed)
    root = smoke_root / scale.scale
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = args.device
    seqs = [TOY[i: i + args.seq_len] for i in range(0, args.n_seqs * args.seq_len, args.seq_len)]
    seqs = [s for s in seqs if len(s) >= 8][: args.n_seqs] or [TOY[: args.seq_len]]

    tok = load_tokenizer(scale.model_id)
    model = load_mlm(scale.model_id, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    hidden = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers)
    assert hidden == scale.embed_dim, f"config says width {scale.embed_dim}, model has {hidden}"
    assert n_layers == scale.num_layers, f"config says {scale.num_layers} layers, model has {n_layers}"

    ppl = pseudo_perplexity(model, tok, seqs, device, 4, 128, masks_per_pass=8).ppl

    # task blocking (inner adversary + outer update) -> B
    block_cfg = OmegaConf.create({"inner_steps_K": 2, "eta_inner": 1e-4, "eta_outer": 5e-5,
                                  "lambda_tr": 1.0, "forget_margin": 6.0, "outer_steps": 2,
                                  "retain_batch": 2, "forget_batch": 2, "warmup_steps": 1,
                                  "mlm_probability": 0.15, "grad_clip": 1.0, "log_every": 1})
    out = train_taskblock(model, tok, seqs, seqs, block_cfg, device, seed=seed, max_length=128)
    save_model(out["model"], str(root / "checkpoints" / "B"))
    b_flops = float(out["flops"]["total_flops"])

    # FLOP-matched control -> A'
    base2 = load_mlm(scale.model_id, device=device)
    ctrl = continued_pretrain(base2, tok, seqs, device, lr=5e-5, batch_size=2, max_length=128,
                              target_flops=b_flops, max_steps=4, seed=seed)
    save_model(ctrl["model"], str(root / "checkpoints" / "A_prime"))

    # attacker cell (fresh reload from the saved B, as the real sweep does)
    atk_cfg = OmegaConf.create({"mode": "full_ft", "lr": 1e-5, "batch_size": 2,
                                "mlm_probability": 0.15, "max_epochs": 1, "weight_decay": 0.0,
                                "patience": 0})
    run_attack(str(root / "checkpoints" / "B"), tok, seqs, seqs, atk_cfg, device, seed=seed,
               max_length=128, ppl_batch_size=4, eval_masks_per_pass=8, eval_every=1)

    # mechanism diagnostics: displacement + a 3-point LMC walk
    base3 = load_mlm(scale.model_id, device=device)
    sd0 = model_state_dict_cpu(base3)
    from src.models.load import load_checkpoint
    sdB = model_state_dict_cpu(load_checkpoint(str(root / "checkpoints" / "B"), device="cpu"))
    disp = l2_by_group(sd0, sdB)["__total__"]["rel"]
    lmc = lmc_path(base3, sd0, sdB,
                   {"f": lambda mo: pseudo_perplexity(mo, tok, seqs, device, 4, 128, masks_per_pass=8).ppl},
                   [0.0, 0.5, 1.0])
    del model, base2, base3, sd0, sdB
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "scale": scale.scale, "model_id": scale.model_id,
        "n_params_actual": n_params, "num_layers_actual": n_layers, "embed_dim_actual": hidden,
        "toy_ppl": float(ppl), "toy_disp_rel": float(disp),
        "toy_lmc_ok": bool(len(lmc["f_ppl"]) == 3 and all(p > 0 for p in lmc["f_ppl"])),
        "toy_block_flops": b_flops, "toy_control_steps": ctrl["steps"],
        "device": args.device,
    }


if __name__ == "__main__":
    main()
