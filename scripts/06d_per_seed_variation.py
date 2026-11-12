"""Phase 7d — seed-level variation for the core endpoint.

`06_analyze.py` medians the acquisition curve over seeds and then reads one crossing
off it, so `multipliers.csv` carries a point estimate with no visible spread. Because
the headline multiplier sits near 1, that spread is exactly what a reader needs: it
decides whether M≈1 is a measurement or a coin flip.

This recomputes the crossing from EACH seed's own curve. The ratio stays paired — the
seed indexes both the attacker's subsample and the training seed, and A/B/A' consume
the same one, so B and A' face an identical attack and the data draw cancels. Reports
every seed value, the median and the min–max range, per condition.

    python scripts/06d_per_seed_variation.py model=esm2_35m paths.root=runs/<run>

Pure pandas + the shared crossing math; no GPU. Writes
results/{per_seed_multipliers.csv, per_seed_summary.csv, per_seed_variation.json} and
results/plots/fig_per_seed_multipliers.{png,pdf}.
"""
import glob
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 7d — PER-SEED VARIATION OF THE MULTIPLIER")
    import pandas as pd

    from src.analysis import plots
    from src.analysis.multiplier import per_seed_multipliers, per_seed_spread
    from src.utils.io import ensure_dir, load_json, save_json

    results = cfg.paths.results
    files = sorted(glob.glob(os.path.join(results, "sweep", "*.csv")))
    if not files:
        raise SystemExit("No sweep CSVs found. Run Phase 6 (05_full_sweep.py) first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["model", "recipe", "lora_rank", "budget_n", "seed", "adversary_data", "target_family"]
    df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    # The relearning probe rides the same directory but is a different endpoint
    # (no crossing is expected at all), so it is not part of the durability table.
    df = df[df["adversary_data"] != "retain_only"]

    tau = float(load_json(os.path.join(results, "thresholds.json"))["tau"])
    per_seed = per_seed_multipliers(df, tau)
    per_seed.to_csv(os.path.join(results, "per_seed_multipliers.csv"), index=False)

    spread = per_seed_spread(per_seed)
    spread.to_csv(os.path.join(results, "per_seed_summary.csv"), index=False)

    # Pooled view over every uncensored condition × seed point on F.
    on_F = per_seed[per_seed["target_family"] == "F"]["mult_B_over_Aprime"].dropna()
    thr = cfg.metrics
    summary = {
        "tau": tau,
        "n_seeds": int(per_seed["seed"].nunique()),
        "n_paired_points_F": int(len(on_F)),
        "median_F": float(on_F.median()) if len(on_F) else None,
        "min_F": float(on_F.min()) if len(on_F) else None,
        "max_F": float(on_F.max()) if len(on_F) else None,
        "n_meeting_durability_gate": int((on_F >= float(thr.durability_min_multiplier)).sum()),
        "n_meeting_headline_gate": int((on_F >= float(thr.headline_min_multiplier)).sum()),
        "n_below_one": int((on_F < 1.0).sum()),
    }
    save_json(summary, os.path.join(results, "per_seed_variation.json"))

    plot_dir = ensure_dir(os.path.join(results, "plots"))
    out = plots.plot_per_seed_multipliers(per_seed, os.path.join(plot_dir, "fig_per_seed_multipliers.png"))

    _bootstrap.banner("PER-SEED PAIRED MULTIPLIERS  B / A'")
    pd.set_option("display.width", 200)
    print(per_seed.to_string(index=False))
    print("\nper condition (seed values, median, range):")
    print(spread.to_string(index=False))
    print()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nartifacts: {os.path.join(results, 'per_seed_multipliers.csv')}, "
          f"{os.path.join(results, 'per_seed_summary.csv')}, {out}")


if __name__ == "__main__":
    main()
