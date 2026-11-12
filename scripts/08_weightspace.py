"""Phase 8 — weight-space mechanism analysis.

Explains *why* the large static forget (B's n=0 forget PPL ≈ 2,989× base) has ~no
adaptation-cost durability, by measuring B directly in weight space:

  1. per-layer ‖θ_B − θ₀‖ (absolute + relative), with A′ as a control;
  2. the linear-mode-connectivity (LMC) barrier on the θ₀ → θ_B path, scoring F-test and
     benign pseudo-perplexity at each interpolation α.

Prediction: small/comparable L2 and ≈0 barrier ⇒ B is linearly connected to base ⇒ the
block is a thin, trivially-reversible perturbation (the Siddiqui et al. LMC story, in the
protein-LM modality).

    python scripts/08_weightspace.py model=esm2_35m

Reads checkpoints/{B,A_prime} and data/splits/; writes results/weightspace.json + two plots.
"""
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 8 — WEIGHT-SPACE MECHANISM (per-layer L2 + LMC barrier)")
    import gc

    import torch

    from src.analysis import plots
    from src.analysis.weightspace import l2_by_group, lmc_path, model_state_dict_cpu
    from src.data.datamodule import DataPaths, subsample, target_test_seqs
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.eval.utility import benign_pseudo_perplexity
    from src.models.load import load_checkpoint, load_tokenizer
    from src.utils.io import ensure_dir, fasta_seqs, save_json

    paths = DataPaths(cfg.paths.data)
    results = cfg.paths.results
    device = cfg.device
    max_len = int(cfg.data.max_length)
    eval_bs = int(cfg.eval.batch_size)
    eval_mpp = int(cfg.eval.masks_per_pass)
    n_eval = int(cfg.eval.max_eval_seqs)
    alphas = list(cfg.weightspace.alphas)
    include_aprime = bool(cfg.weightspace.get("include_aprime", True))

    tok = load_tokenizer(cfg.model.name)
    f_test = subsample(target_test_seqs(paths, "F"), n_eval, seed=cfg.seed)
    holdout = subsample(fasta_seqs(paths.retain_holdout), n_eval, seed=cfg.seed)

    def _free():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- per-layer L2 (B vs base, A' vs base) ------------------------------ #
    base = load_checkpoint(cfg.model.name, device=device)
    sd0 = model_state_dict_cpu(base)

    B = load_checkpoint(os.path.join(cfg.paths.checkpoints, "B"), device=device)
    sdB = model_state_dict_cpu(B)
    del B
    _free()
    l2_B = l2_by_group(sd0, sdB)
    print(f"‖θ_B − θ₀‖ total: rel={l2_B['__total__']['rel']:.4f}  abs={l2_B['__total__']['l2']:.3f}")

    l2_Ap = None
    if include_aprime:
        Ap = load_checkpoint(os.path.join(cfg.paths.checkpoints, "A_prime"), device=device)
        l2_Ap = l2_by_group(sd0, model_state_dict_cpu(Ap))
        del Ap
        _free()
        print(f"‖θ_A′ − θ₀‖ total: rel={l2_Ap['__total__']['rel']:.4f}  abs={l2_Ap['__total__']['l2']:.3f}")

    # ---- LMC barrier on θ₀ → θ_B ------------------------------------------- #
    def f_fn(m):
        return pseudo_perplexity(m, tok, f_test, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl

    def benign_fn(m):
        return benign_pseudo_perplexity(m, tok, holdout, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl

    print(f"LMC over {len(alphas)} α (θ₀ → θ_B), scoring F-test + benign PPL ...")
    lmc = lmc_path(base, sd0, sdB, {"f": f_fn, "benign": benign_fn}, alphas)

    record = {
        "l2": {"B": l2_B, **({"A_prime": l2_Ap} if l2_Ap else {})},
        "lmc_B": lmc,
        "config": {"alphas": alphas, "max_eval_seqs": n_eval},
    }
    save_json(record, os.path.join(results, "weightspace.json"))

    plot_dir = ensure_dir(os.path.join(results, "plots"))
    plots.plot_layer_l2(l2_B, l2_Ap, os.path.join(plot_dir, "fig_weightspace_l2.png"))
    plots.plot_lmc_barrier(lmc, os.path.join(plot_dir, "fig_lmc_barrier.png"))

    # ---- report ------------------------------------------------------------ #
    _bootstrap.banner("WEIGHT-SPACE MECHANISM")
    line = f"  total relative ‖Δθ‖:  B = {l2_B['__total__']['rel']:.4f}"
    if l2_Ap:
        line += f"   A′ = {l2_Ap['__total__']['rel']:.4f}   (B/A′ = {l2_B['__total__']['rel']/l2_Ap['__total__']['rel']:.2f}×)"
    print(line)
    print(f"  LMC θ₀→θ_B  F-test PPL: {lmc['f_ppl'][0]:.2f} → {lmc['f_ppl'][-1]:.1f}   "
          f"barrier = {lmc['f_barrier_nats']:.3f} nats   monotonic = {lmc['f_monotonic']}")
    print(f"  LMC θ₀→θ_B  benign PPL: {lmc['benign_ppl'][0]:.2f} → {lmc['benign_ppl'][-1]:.2f}   "
          f"barrier = {lmc['benign_barrier_nats']:.3f} nats")
    # "no barrier" ⇔ F path monotonic (no interior ridge) AND benign flat — not an
    # absolute nat threshold (the endpoints differ by ~8 nats, so a small convex bulge
    # on a monotonic climb is not a ridge).
    no_barrier = lmc["f_monotonic"] and lmc["benign_barrier_nats"] < 0.5
    verdict = ("linearly connected to base (no ridge: F monotonic, benign flat) — thin, "
               "reversible perturbation" if no_barrier else "a barrier exists on the θ₀→θ_B path")
    print(f"  => {verdict}")
    print(f"\nartifacts: {os.path.join(results, 'weightspace.json')}, "
          f"{plot_dir}/fig_weightspace_l2.png, {plot_dir}/fig_lmc_barrier.png")


if __name__ == "__main__":
    main()
