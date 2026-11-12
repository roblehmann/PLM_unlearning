"""Phase 6b (CIR) — efficacy-vs-disruption frontier collector + mechanism readout.

Aggregates the CIR operating-point sweep from slurm/03c_cir_frontier.sbatch: each cell
under results/cir_frontier/<tag>/ trained a B at a given (unlearn_rate,
terminate_retain_ppl_ratio). For every cell this reads B_compute.json (forget-deletion
ratio + benign drift) AND computes the kept-vs-collapsed weight displacement of
ΔW = θ_B − θ₀ from the saved collapse bases (the CIR mechanism readout, B vs base) — so
you see, at each operating point, both *how much* F was deleted and *whether* the move
lives in the family-relevant (kept) subspace. Writes results/cir_frontier/frontier.csv
(+ a plot) and flags the best deletion within the 0.1% disruption budget.

    python scripts/06b_cir_frontier.py model=esm2_8m      # after the frontier array job

Pure weight-space math + checkpoint loads (no GPU forward passes).
"""
import glob
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 6b (CIR) — EFFICACY vs DISRUPTION FRONTIER")
    import numpy as np
    import pandas as pd

    from src.analysis.weightspace import model_state_dict_cpu, projected_l2_by_group
    from src.models.load import load_checkpoint
    from src.utils.io import ensure_dir, load_json

    frontier_name = cfg.get("frontier_dir", "cir_frontier")  # +frontier_dir=cir_frontier_fine to collect the fine sweep
    fr_results = os.path.join(cfg.paths.results, frontier_name)
    fr_ckpts = os.path.join(cfg.paths.checkpoints, frontier_name)
    cells = sorted(glob.glob(os.path.join(fr_results, "*", "B_compute.json")))
    if not cells:
        raise SystemExit(f"No cells under {fr_results}/*/B_compute.json — run slurm/03c_cir_frontier.sbatch first.")

    sd0 = model_state_dict_cpu(load_checkpoint(cfg.model.name, device="cpu"))  # base θ₀, loaded once
    rows = []
    for cell in cells:
        tag = os.path.basename(os.path.dirname(cell))
        rec = load_json(cell)
        row = {
            "tag": tag,
            "unlearn_rate": rec.get("unlearn_rate"),
            "terminate_ratio": rec.get("terminate_retain_ppl_ratio"),
            "benign_drift_rel": rec.get("benign_drift_rel"),
            "forget_ratio": rec.get("forget_ppl_ratio_B_over_base"),
            "forget_ppl_B": rec.get("forget_ppl_B"),
            "epochs_run": rec.get("epochs_run"),
            "optimizer_steps": rec.get("optimizer_steps"),
        }
        # mechanism readout: split ΔW = θ_B − θ₀ into kept vs collapsed subspaces
        npz_p = os.path.join(fr_results, tag, "cir_subspace.npz")
        ckpt_b = os.path.join(fr_ckpts, tag, "B")
        if os.path.exists(npz_p) and os.path.isdir(ckpt_b):
            d = np.load(npz_p)
            bases = {int(li): {"act_basis": d[f"act_basis_{li}"], "grad_basis": d[f"grad_basis_{li}"]}
                     for li in d["layers"]}
            sdB = model_state_dict_cpu(load_checkpoint(ckpt_b, device="cpu"))
            tot = projected_l2_by_group(sd0, sdB, bases)["__total__"]
            row["kept_l2"] = tot["kept_l2"]
            row["collapsed_l2"] = tot["collapsed_l2"]
            row["kept_frac"] = tot["kept_l2"] / (tot["total_l2"] + 1e-12)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("benign_drift_rel").reset_index(drop=True)
    ensure_dir(fr_results)
    csv = os.path.join(fr_results, "frontier.csv")
    df.to_csv(csv, index=False)

    budget = 0.001  # the 0.1% retain-disruption budget
    within = df[df["benign_drift_rel"] <= budget].sort_values("forget_ratio", ascending=False)

    _bootstrap.banner("CIR FRONTIER (sorted by benign drift)")
    print(df.to_string(index=False))
    print(f"\nbest deletion within the {budget*100:.1f}% disruption budget:")
    if len(within):
        b = within.iloc[0]
        extra = f"   kept/collapsed ΔW = {b['kept_l2']:.3f}/{b['collapsed_l2']:.3f}" if "kept_l2" in b and pd.notna(b.get("kept_l2")) else ""
        print(f"  {b['tag']}: forget {b['forget_ratio']:.2f}x  @ drift {b['benign_drift_rel']*100:.3f}%{extra}")
    else:
        print(f"  none stayed within {budget*100:.1f}% (all overshot) — lower unlearn_rate or steps_per_epoch "
              "for finer budget control, or read the lowest-drift row above.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 4))
        sc = ax.scatter(df["benign_drift_rel"] * 100, df["forget_ratio"],
                        c=df["unlearn_rate"], cmap="viridis", s=45, zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r["tag"], (r["benign_drift_rel"] * 100, r["forget_ratio"]), fontsize=6, alpha=0.7)
        ax.axvline(budget * 100, ls="--", c="r", lw=1, label="0.1% budget")
        ax.axhline(2.0, ls=":", c="grey", lw=1, label="2x = UNLEARNED")
        ax.set_xlabel("benign drift (%)")
        ax.set_ylabel("forget PPL ratio  B / base")
        ax.set_xscale("log")
        ax.legend(fontsize=7)
        fig.colorbar(sc, label="unlearn_rate")
        ax.set_title("CIR efficacy-vs-disruption frontier")
        fig.tight_layout()
        png = os.path.join(fr_results, "frontier.png")
        fig.savefig(png, dpi=130)
        print(f"\nplot  -> {png}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e!r})")
    print(f"table -> {csv}")


if __name__ == "__main__":
    main()
