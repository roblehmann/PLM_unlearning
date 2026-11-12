"""Phase 7 — analysis.

Computes the adaptation-cost multiplier (B/A' headline, B/A secondary, A'/A
validity, with right-censoring), the utility-vs-cost frontier, and the durability
table; applies the thresholds.yaml gates; prints a pass/fail summary against the
pre-registered claim.

    python scripts/06_analyze.py model=esm2_35m

Done when: plots + results table + pass/fail summary are produced; censoring and the
A'/A validity check are reported.
"""
import glob
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 7 — ANALYSIS")
    import pandas as pd
    from omegaconf import OmegaConf

    from src.analysis import plots
    from src.analysis.multiplier import compute_multipliers, pass_fail_summary
    from src.data.datamodule import DataPaths, subsample, target_test_seqs
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.eval.utility import benign_pseudo_perplexity, utility_retention
    from src.models.load import load_checkpoint, load_tokenizer
    from src.utils.io import ensure_dir, fasta_seqs, load_json, save_json

    paths = DataPaths(cfg.paths.data)
    results = cfg.paths.results
    device = cfg.device
    max_len = int(cfg.data.max_length)

    files = sorted(glob.glob(os.path.join(results, "sweep", "*.csv")))
    if not files:
        raise SystemExit("No sweep CSVs found. Run Phase 6 (05_full_sweep.py) first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Resume re-runs can write a cell more than once across shard files; keep one row
    # per unique cell so medians aren't double-weighted.
    key = ["model", "recipe", "lora_rank", "budget_n", "seed", "adversary_data", "target_family"]
    n_before = len(df)
    df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    print(f"loaded {n_before} sweep rows from {len(files)} shard file(s); "
          f"{len(df)} unique cells after de-duplication")

    thr = load_json(os.path.join(results, "thresholds.json"))
    tau, max_budget = float(thr["tau"]), float(thr["max_budget"])

    mult = compute_multipliers(df, tau, max_budget)
    mult.to_csv(os.path.join(results, "multipliers.csv"), index=False)

    # ---- benign utility (B vs A') + n=0 forget PPL (failure localization) --- #
    # The n=0 forget PPL = each checkpoint's pseudo-perplexity on F_test with NO
    # attacker fine-tuning. It localizes a low multiplier: if B's n=0 forget PPL is
    # high (>> A) the defender DID unlearn F and the failure is the attacker
    # re-learning; if it's ~A, the block never fired (defender didn't unlearn).
    import gc

    import torch

    ppl_B = ppl_Ap = util = None
    forget_n0 = {}
    eval_bs = int(cfg.eval.batch_size)
    eval_mpp = int(cfg.eval.masks_per_pass)
    n_eval = int(cfg.eval.max_eval_seqs)
    holdout = subsample(fasta_seqs(paths.retain_holdout), n_eval, seed=cfg.seed)
    f_test = subsample(target_test_seqs(paths, "F"), n_eval, seed=cfg.seed)
    tok = load_tokenizer(cfg.model.name)

    def _free(model):
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # A = base ESM-2 (its n=0 forget PPL is the pre-attack PPL on F_test).
    ckpts = {"A": cfg.model.name,
             "B": os.path.join(cfg.paths.checkpoints, "B"),
             "A_prime": os.path.join(cfg.paths.checkpoints, "A_prime")}
    try:
        for name, ckpt in ckpts.items():
            m = load_checkpoint(ckpt, device=device)
            forget_n0[name] = pseudo_perplexity(m, tok, f_test, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
            if name == "B":
                ppl_B = benign_pseudo_perplexity(m, tok, holdout, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
            elif name == "A_prime":
                ppl_Ap = benign_pseudo_perplexity(m, tok, holdout, device, eval_bs, max_len, masks_per_pass=eval_mpp).ppl
            print(f"[analyze] {name}: n=0 forget PPL (F_test) = {forget_n0[name]:.4f}")
            _free(m)
        if ppl_B is not None and ppl_Ap is not None:
            util = utility_retention(ppl_B, ppl_Ap)
    except Exception as e:  # noqa: BLE001
        print(f"[analyze] checkpoint evals skipped (missing?): {e!r}")

    summary = pass_fail_summary(mult, util, OmegaConf.to_container(cfg.metrics, resolve=True))
    summary.update({"tau": tau, "benign_ppl_B": ppl_B, "benign_ppl_Aprime": ppl_Ap, "n_sweep_rows": len(df)})
    # n=0 forget PPL + failure localization
    if forget_n0:
        summary["forget_ppl_n0"] = forget_n0
        fA, fB = forget_n0.get("A"), forget_n0.get("B")
        if fA and fB:
            summary["n0_forget_ratio_B_over_A"] = fB / fA
            summary["failure_localization"] = (
                "defender_unlearned__failure_is_attacker_relearning" if fB / fA >= 2.0
                else "defender_did_not_unlearn__block_never_fired"
            )
    save_json(summary, os.path.join(results, "summary.json"))

    # ---- plots ------------------------------------------------------------- #
    plot_dir = ensure_dir(os.path.join(results, "plots"))
    plots.plot_all_conditions(df, tau, plot_dir)
    if ppl_B is not None and ppl_Ap is not None:
        plots.plot_utility_frontier(mult, {"B": ppl_B, "A_prime": ppl_Ap},
                                    os.path.join(plot_dir, "frontier.png"),
                                    util_tol=float(cfg.metrics.utility_tolerance))
    plots.plot_durability(mult, os.path.join(plot_dir, "durability.png"))

    # ---- report ------------------------------------------------------------ #
    _bootstrap.banner("PASS/FAIL vs PRE-REGISTERED CLAIM")
    print(mult.to_string(index=False))
    if forget_n0:
        print("\nn=0 forget PPL on F_test (no attacker fine-tuning):")
        for name in ("A", "A_prime", "B"):
            if name in forget_n0:
                print(f"  {name:8s} {forget_n0[name]:.4f}")
        if "failure_localization" in summary:
            print(f"  -> {summary['failure_localization']} "
                  f"(B/A = {summary.get('n0_forget_ratio_B_over_A', float('nan')):.1f}x)")
    print()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nartifacts: {os.path.join(results, 'multipliers.csv')}, "
          f"{os.path.join(results, 'summary.json')}, {plot_dir}/")


if __name__ == "__main__":
    main()
