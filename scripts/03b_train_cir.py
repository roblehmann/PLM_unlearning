"""Phase 4 (CIR variant) — CIR unlearning trainer -> checkpoint B.

A parallel to scripts/03_train_blocked.py that runs CIR (src/train/cir.py) instead of
first-order task blocking. It writes the SAME artifacts (checkpoints/B,
results/B_compute.json) so the compute-matched control (Phase 5) and the attacker
sweep / analysis / weight-space phases (6–8) are completely unchanged. It additionally
saves the per-layer collapse subspace bases to results/cir_subspace.npz for the
Phase-8b subspace-projected-displacement probe (scripts/08b_cir_subspace.py).

    python scripts/03b_train_cir.py model=esm2_8m blocking=cir blocking.max_epochs=1 \
        blocking.pc_corpus_seqs=64 device=cpu tracking.backend=none           # smoke
    python scripts/03b_train_cir.py model=esm2_35m blocking=cir               # headline (Option A)
    python scripts/03b_train_cir.py model=esm2_35m blocking=cir blocking.subspace_mode=forget_pca  # ablation B

NOTE: this overwrites checkpoints/B + results/B_compute.json from any prior (MLAC) run.
Archive first if you want to keep both:
    cp -r checkpoints/B checkpoints/B_mlac ; cp results/B_compute.json results/B_compute_mlac.json
"""
from pathlib import Path

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 4 (CIR) — TRAIN CIR-UNLEARNED MODEL B")
    import numpy as np

    from src.data.datamodule import DataPaths, subsample
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.eval.utility import benign_pseudo_perplexity, utility_retention
    from src.models.load import load_mlm, load_tokenizer, save_model
    from src.train.cir import train_cir
    from src.train.mlm import cfg_get
    from src.utils.io import ensure_dir, fasta_seqs, save_json
    from src.utils.seeding import set_seed

    set_seed(cfg.seed)
    paths = DataPaths(cfg.paths.data)
    device = cfg.device
    max_len = int(cfg.data.max_length)
    ckpt_B = str(Path(cfg.paths.checkpoints) / "B")
    tracker = _bootstrap.init_tracker_from_cfg(cfg, run_name="train_cir")

    eval_bs = int(cfg.eval.batch_size)
    eval_mpp = int(cfg.eval.masks_per_pass)
    n_eval = int(cfg.eval.max_eval_seqs)
    tok = load_tokenizer(cfg.model.name)
    retain = fasta_seqs(paths.retain_train)
    forget = fasta_seqs(paths.F_train)
    holdout = subsample(fasta_seqs(paths.retain_holdout), n_eval, seed=cfg.seed)
    f_test = subsample(fasta_seqs(paths.F_test), n_eval, seed=cfg.seed)
    print(f"retain_train={len(retain)} forget(F_train)={len(forget)} benign-eval holdout={len(holdout)} F_test={len(f_test)}")
    print(f"subspace_mode={cfg.blocking.subspace_mode}  layer_band={list(cfg.blocking.layer_band)}")

    # benign + forget PPL of the base BEFORE CIR (utility-drift + did-we-actually-unlearn checks)
    base = load_mlm(cfg.model.name, device=device)
    ppl_base = benign_pseudo_perplexity(base, tok, holdout, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
    forget_ppl_base = pseudo_perplexity(base, tok, f_test, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
    print(f"benign PPL (base, retain holdout) = {ppl_base:.4f}   forget PPL (base, F_test) = {forget_ppl_base:.4f}")

    out = train_cir(
        base, tok, retain, forget, cfg.blocking, device,
        seed=cfg.seed, max_length=max_len, tracker=tracker,
        retain_eval_seqs=holdout, eval_batch_size=eval_bs,
        eval_masks_per_pass=eval_mpp, eval_max_seqs=n_eval,
    )
    ensure_dir(cfg.paths.checkpoints)
    save_model(out["model"], ckpt_B)

    # per-layer collapse bases -> npz (for Phase-8b subspace-projected displacement)
    ensure_dir(cfg.paths.results)
    npz = {}
    for li, b in out["subspace"].items():
        for key, arr in b.items():
            npz[f"{key}_{li}"] = arr
    npz["layers"] = np.array(out["layers"], dtype=np.int64)
    np.savez(f"{cfg.paths.results}/cir_subspace.npz", **npz)

    ppl_B = benign_pseudo_perplexity(out["model"], tok, holdout, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
    drift = utility_retention(ppl_B, ppl_base)
    forget_ppl_B = pseudo_perplexity(out["model"], tok, f_test, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
    forget_ratio = forget_ppl_B / forget_ppl_base if forget_ppl_base > 0 else float("nan")
    record = {
        "checkpoint": ckpt_B,
        "method": "cir",
        "subspace_mode": str(cfg.blocking.subspace_mode),
        "unlearn_rate": float(cfg_get(cfg.blocking, "unlearn_rate", float("nan"))),
        "terminate_retain_ppl_ratio": float(cfg_get(cfg.blocking, "terminate_retain_ppl_ratio", float("nan"))),
        "steps_per_epoch": int(cfg_get(cfg.blocking, "steps_per_epoch", 0)),
        "n_act_pcs": int(cfg_get(cfg.blocking, "n_act_pcs", 0)),
        "n_grad_pcs": int(cfg_get(cfg.blocking, "n_grad_pcs", 0)),
        "flops_total": out["flops"]["total_flops"],
        "optimizer_steps": out["flops"]["optimizer_steps"],
        "overhead_factor_vs_plain_step": out["flops"]["overhead_factor"],
        "wall_clock_s": out["flops"]["wall_clock_s"],
        "benign_ppl_base": ppl_base,
        "benign_ppl_B": ppl_B,
        "benign_drift_rel": drift,
        "forget_ppl_base": forget_ppl_base,
        "forget_ppl_B": forget_ppl_B,
        "forget_ppl_ratio_B_over_base": forget_ratio,
        "stopped_early": bool(out["stopped_early"]),
        "epochs_run": len(out["history"]),
    }
    save_json(record, f"{cfg.paths.results}/B_compute.json")
    tracker.log_summary(record)
    tracker.finish()

    _bootstrap.banner("CIR MODEL B DONE")
    print(f"  defender FLOPs={record['flops_total']:.3e}  overhead≈{record['overhead_factor_vs_plain_step']:.1f}x "
          f"plain step  wall={record['wall_clock_s']:.0f}s  epochs={record['epochs_run']}  early_stop={record['stopped_early']}")
    print(f"  benign PPL drift (B vs base) = {drift*100:.2f}%  -> {'OK' if drift < 0.05 else 'CHECK unlearn_rate/layer_band'}")
    print(f"  forget PPL (F_test): base {forget_ppl_base:.3f} -> B {forget_ppl_B:.3f}  (B/base = {forget_ratio:.2f}x)"
          f"  -> {'UNLEARNED' if forget_ratio >= 2.0 else 'little/no deletion (raise unlearn_rate)'}")
    print(f"  subspace bases -> {cfg.paths.results}/cir_subspace.npz")


if __name__ == "__main__":
    main()
