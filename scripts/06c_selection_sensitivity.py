"""Selection-rule sensitivity: is the multiplier an artifact of test-set early stopping?

The harness scores held-out F_test every `eval_every` epochs, early-stops on a patience
counter over THAT curve, and reports `best_target_ppl` = min over evaluated epochs
(src/train/attack.py). Both the stopping decision and the reported value therefore select
on the same held-out set the multiplier is read from — an optimistic (test-selected) rule.

This script re-derives every crossing budget and multiplier under a TEST-INDEPENDENT
rule instead: the endpoint of the pre-declared schedule (`final_target_ppl`, i.e. the PPL
at the last epoch actually run, with no min-over-epochs), and reports the delta. It also
reports how often the two rules coincide (best_epoch == last epoch => no selection
happened at all) and the optimism gap (final - best).

No GPU, no checkpoints needed — reads the sweep CSVs + thresholds.json only.

    python scripts/06c_selection_sensitivity.py runs/<run>/results
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis.multiplier import compute_multipliers  # noqa: E402

KEY = ["model", "recipe", "lora_rank", "budget_n", "seed", "adversary_data", "target_family"]


def load_sweep(results: str, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results, "sweep", pattern)))
    if not files:
        raise SystemExit(f"no sweep CSVs matching {pattern!r} under {results}/sweep")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)


def main() -> None:
    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*.csv"
    df = load_sweep(results, pattern)
    thr = json.load(open(os.path.join(results, "thresholds.json")))
    tau, max_budget = float(thr["tau"]), float(thr["max_budget"])

    # --- how much selection actually happened -------------------------------- #
    gap = (df["final_target_ppl"] - df["best_target_ppl"]).dropna()
    no_selection = int((df["best_epoch"] == df["epochs_run"] * 0 + df["best_epoch"].max()).sum())
    stats = {
        "n_cells": int(len(df)),
        "n_cells_best_is_last_eval": int((gap.abs() < 1e-9).sum()),
        "frac_cells_no_selection": float((gap.abs() < 1e-9).mean()),
        "optimism_gap_ppl_median": float(gap.median()),
        "optimism_gap_ppl_mean": float(gap.mean()),
        "optimism_gap_ppl_max": float(gap.max()),
        "optimism_gap_rel_max": float((gap / df["best_target_ppl"]).max()),
        "best_epoch_counts": {str(k): int(v) for k, v in df["best_epoch"].value_counts().sort_index().items()},
    }

    # --- tau is itself a min-over-epochs of the floor fit --------------------- #
    fh = thr.get("floor_history") or []
    if fh:
        stats["floor_ppl_selected_min"] = float(thr["ppl_floor"])
        stats["floor_ppl_at_final_epoch"] = float(fh[-1]["target_ppl"])
        stats["floor_monotone_to_end"] = bool(
            all(fh[i + 1]["target_ppl"] < fh[i]["target_ppl"] for i in range(len(fh) - 1))
        )
        stats["tau_is_selection_free"] = bool(
            abs(stats["floor_ppl_selected_min"] - stats["floor_ppl_at_final_epoch"]) < 1e-9
        )

    # --- re-derive multipliers under both rules ------------------------------- #
    frames = {}
    for col in ("best_target_ppl", "final_target_ppl"):
        m = compute_multipliers(df, tau, max_budget, ppl_col=col)
        frames[col] = m.set_index([c for c in ("recipe", "lora_rank", "adversary_data", "target_family")
                                   if c in m.columns])
    b, f = frames["best_target_ppl"], frames["final_target_ppl"]
    cmp = pd.DataFrame({
        "n_cross_A_selected": b["n_cross_A"], "n_cross_A_fixed": f["n_cross_A"],
        "n_cross_B_selected": b["n_cross_B"], "n_cross_B_fixed": f["n_cross_B"],
        "n_cross_Aprime_selected": b["n_cross_A_prime"], "n_cross_Aprime_fixed": f["n_cross_A_prime"],
        "M_selected": b["mult_B_over_Aprime"], "M_fixed": f["mult_B_over_Aprime"],
        "validity_selected": b["mult_Aprime_over_A"], "validity_fixed": f["mult_Aprime_over_A"],
    })
    cmp["M_delta_pct"] = 100.0 * (cmp["M_fixed"] - cmp["M_selected"]) / cmp["M_selected"]

    out_csv = os.path.join(results, "selection_sensitivity.csv")
    out_json = os.path.join(results, "selection_sensitivity.json")
    cmp.reset_index().to_csv(out_csv, index=False)
    finite = cmp["M_delta_pct"].dropna()
    stats["M_delta_pct_max_abs"] = float(finite.abs().max()) if len(finite) else None
    stats["tau"] = tau
    json.dump(stats, open(out_json, "w"), indent=2)

    print(f"cells={stats['n_cells']}  no-selection cells={stats['n_cells_best_is_last_eval']} "
          f"({100*stats['frac_cells_no_selection']:.1f}%)")
    print(f"optimism gap (final-best) PPL: median={stats['optimism_gap_ppl_median']:.4f} "
          f"mean={stats['optimism_gap_ppl_mean']:.4f} max={stats['optimism_gap_ppl_max']:.4f} "
          f"({100*stats['optimism_gap_rel_max']:.2f}% rel)")
    if fh:
        print(f"tau selection-free (floor min == floor at final epoch): {stats['tau_is_selection_free']}")
    print(cmp.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nmax |ΔM| under the test-independent rule: {stats['M_delta_pct_max_abs']:.2f}%")
    print(f"wrote {out_csv}\n      {out_json}")


if __name__ == "__main__":
    main()
