"""Phase 4 — task-blocking trainer -> checkpoint B.

Debug on the 8M model first (`model=esm2_8m`), then run on 35M from the base
checkpoint to produce B. Records defender-side compute (FLOPs, overhead vs a plain
MLM step, wall-clock) as the reference cost figure.

    python scripts/03_train_blocked.py model=esm2_8m blocking.outer_steps=50   # debug
    python scripts/03_train_blocked.py model=esm2_35m                          # headline

Done when: B exists; training curves logged; B benign held-out PPL within a few %
of base; defender-side overhead recorded.
"""
from pathlib import Path

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 4 — TRAIN TASK-BLOCKED MODEL B")
    from src.data.datamodule import DataPaths, subsample
    from src.eval.utility import benign_pseudo_perplexity, utility_retention
    from src.models.load import load_mlm, load_tokenizer, save_model
    from src.train.taskblock import train_taskblock
    from src.utils.io import ensure_dir, fasta_seqs, save_json
    from src.utils.seeding import set_seed

    set_seed(cfg.seed)
    paths = DataPaths(cfg.paths.data)
    device = cfg.device
    max_len = int(cfg.data.max_length)
    ckpt_B = str(Path(cfg.paths.checkpoints) / "B")
    tracker = _bootstrap.init_tracker_from_cfg(cfg, run_name="train_blocked")

    eval_bs = int(cfg.eval.batch_size)
    eval_mpp = int(cfg.eval.masks_per_pass)
    tok = load_tokenizer(cfg.model.name)
    retain = fasta_seqs(paths.retain_train)
    forget = fasta_seqs(paths.F_train)
    holdout = subsample(fasta_seqs(paths.retain_holdout), int(cfg.eval.max_eval_seqs), seed=cfg.seed)
    print(f"retain_train={len(retain)} forget(F_train)={len(forget)} benign-eval holdout={len(holdout)}")

    # benign PPL of the base BEFORE blocking (for the utility-drift check)
    base = load_mlm(cfg.model.name, device=device)
    ppl_base = benign_pseudo_perplexity(base, tok, holdout, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
    print(f"benign PPL (base, retain holdout) = {ppl_base:.4f}")

    out = train_taskblock(base, tok, retain, forget, cfg.blocking, device,
                          seed=cfg.seed, max_length=max_len, tracker=tracker)
    ensure_dir(cfg.paths.checkpoints)
    save_model(out["model"], ckpt_B)

    ppl_B = benign_pseudo_perplexity(out["model"], tok, holdout, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
    drift = utility_retention(ppl_B, ppl_base)
    record = {
        "checkpoint": ckpt_B,
        "flops_total": out["flops"]["total_flops"],
        "optimizer_steps": out["flops"]["optimizer_steps"],
        "overhead_factor_vs_plain_step": out["flops"]["overhead_factor"],
        "wall_clock_s": out["flops"]["wall_clock_s"],
        "benign_ppl_base": ppl_base,
        "benign_ppl_B": ppl_B,
        "benign_drift_rel": drift,
    }
    save_json(record, f"{cfg.paths.results}/B_compute.json")
    tracker.log_summary(record)
    tracker.finish()

    _bootstrap.banner("BLOCKED MODEL B DONE")
    print(f"  defender FLOPs={record['flops_total']:.3e}  overhead≈{record['overhead_factor_vs_plain_step']:.1f}x "
          f"plain step  wall={record['wall_clock_s']:.0f}s")
    print(f"  benign PPL drift (B vs base) = {drift*100:.2f}%  -> {'OK' if drift < 0.05 else 'CHECK lambda_tr/forget_margin'}")


if __name__ == "__main__":
    main()
