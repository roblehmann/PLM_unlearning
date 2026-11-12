"""Tests for src.analysis (multiplier with censoring, plots)."""
import pandas as pd
import pytest

from src.analysis.multiplier import (
    Crossing,
    compute_multipliers,
    crossing_budget,
    pass_fail_summary,
    ratio,
)


def test_crossing_interpolated():
    c = crossing_budget([16, 64, 256], [10, 5, 2], tau=4.0)
    assert not c.censored
    assert 64 < c.n < 256  # interpolated between the bracketing budgets


def test_crossing_censored_and_first_point():
    assert crossing_budget([16, 64], [10, 9], tau=4.0).censored
    c0 = crossing_budget([16, 64], [3, 2], tau=4.0)  # already below at the first budget
    assert not c0.censored and c0.n == 16


def test_ratio_censoring():
    den = Crossing(n=32.0, censored=False)
    num_ok = Crossing(n=128.0, censored=False)
    assert ratio(num_ok, den, 256).value == pytest.approx(4.0)
    cens = ratio(Crossing(n=None, censored=True), den, 256)
    assert cens.censored and cens.lower_bound and cens.value == pytest.approx(8.0)  # 256/32
    assert pd.isna(ratio(num_ok, Crossing(n=None, censored=True), 256).value)


def _toy_sweep_df():
    rows = []
    ppl = {"A": [3, 2, 1], "A_prime": [3, 2, 1], "B": [3, 3, 3]}  # B never crosses
    budgets = [16, 64, 256]
    for model in ("A", "B", "A_prime"):
        for b, p in zip(budgets, ppl[model]):
            for seed in (0, 1):
                rows.append({
                    "model": model, "recipe": "lora", "lora_rank": 8,
                    "adversary_data": "clean", "target_family": "F",
                    "budget_n": b, "seed": seed, "best_target_ppl": p,
                })
    return pd.DataFrame(rows)


def test_compute_multipliers_censored_B():
    df = _toy_sweep_df()
    out = compute_multipliers(df, tau=2.5, max_budget=256)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["censored_B"]
    assert row["mult_B_over_Aprime_is_lower_bound"]
    assert row["mult_Aprime_over_A"] == pytest.approx(1.0)   # validity check ≈ 1
    assert row["mult_B_over_Aprime"] > 1.0


def test_pass_fail_summary():
    mult = pd.DataFrame([
        {"target_family": "F", "adversary_data": "clean",
         "mult_B_over_Aprime": 12.0, "mult_B_over_Aprime_censored": False, "mult_Aprime_over_A": 1.0},
        {"target_family": "F", "adversary_data": "homolog_augmented",
         "mult_B_over_Aprime": 4.0, "mult_B_over_Aprime_censored": False, "mult_Aprime_over_A": 1.0},
    ])
    s = pass_fail_summary(mult, utility_retention=0.01,
                          thresholds={"headline_min_multiplier": 10.0, "durability_min_multiplier": 3.0, "utility_tolerance": 0.02})
    assert s["headline_multiplier_pass"] is True
    assert s["durability_multiplier_pass"] is True
    assert s["utility_pass"] is True
    assert s["overall_pass"] is True


def test_plots_smoke(tmp_path):
    from src.analysis import plots

    df = _toy_sweep_df()
    p = plots.plot_acquisition_curves(df, tau=2.5, out_path=tmp_path / "acq.png", title="t")
    assert (tmp_path / "acq.png").exists()
    out = compute_multipliers(df, tau=2.5, max_budget=256)
    dpath = plots.plot_durability(out, tmp_path / "dur.png")
    assert dpath == "" or (tmp_path / "dur.png").exists()


def _seeded_sweep_df():
    """Two seeds whose absolute crossings differ a lot but whose B/A' ratio is the same:
    seed 1's curves are the seed-0 curves shifted to a higher budget. A per-seed paired
    ratio should recover the common ratio from both seeds."""
    rows = []
    budgets = [16, 64, 256, 1024]
    curves = {
        0: {"A": [9, 7, 5, 3], "A_prime": [9, 7, 5, 3], "B": [9, 7.5, 5.8, 3.5]},
        1: {"A": [9, 8, 6, 4], "A_prime": [9, 8, 6, 4], "B": [9, 8.5, 6.5, 4.9]},
    }
    for seed, per_model in curves.items():
        for model, ppls in per_model.items():
            for b, p in zip(budgets, ppls):
                rows.append({"model": model, "recipe": "lora", "lora_rank": 8, "budget_n": b,
                             "seed": seed, "adversary_data": "clean", "target_family": "F",
                             "best_target_ppl": p})
    return pd.DataFrame(rows)


def test_per_seed_multipliers_are_paired_per_seed():
    from src.analysis.multiplier import per_seed_multipliers

    ps = per_seed_multipliers(_seeded_sweep_df(), tau=5.5)
    assert len(ps) == 2                      # one row per condition x seed
    assert set(ps["seed"]) == {0, 1}
    # each seed's crossings come from its own curve, so the harder seed crosses later
    s0 = ps[ps.seed == 0].iloc[0]
    s1 = ps[ps.seed == 1].iloc[0]
    assert s1["n_cross_B"] > s0["n_cross_B"]
    # ...yet both arms shift together, so the paired ratio is stable across seeds
    assert s0["mult_B_over_Aprime"] > 1
    assert s0["mult_B_over_Aprime"] == pytest.approx(s1["mult_B_over_Aprime"], rel=0.05)


def test_per_seed_multipliers_marks_censored_seeds():
    from src.analysis.multiplier import per_seed_multipliers, per_seed_spread

    df = _seeded_sweep_df()
    df.loc[(df.model == "B") & (df.seed == 1), "best_target_ppl"] = 9.0  # never crosses
    ps = per_seed_multipliers(df, tau=5.5)
    censored = ps[ps.seed == 1].iloc[0]
    assert censored["censored_B"] and pd.isna(censored["mult_B_over_Aprime"])

    spread = per_seed_spread(ps).iloc[0]
    assert spread["n_seeds"] == 1 and spread["n_censored"] == 1  # censored seed not averaged in


def test_per_seed_spread_reports_median_and_range():
    from src.analysis.multiplier import per_seed_spread

    ps = pd.DataFrame({
        "recipe": ["lora"] * 3, "lora_rank": [8] * 3, "adversary_data": ["clean"] * 3,
        "target_family": ["F"] * 3, "seed": [0, 1, 2],
        "mult_B_over_Aprime": [0.95, 1.05, 0.99],
    })
    row = per_seed_spread(ps).iloc[0]
    assert row["median"] == pytest.approx(0.99)
    assert (row["min"], row["max"]) == pytest.approx((0.95, 1.05))
    assert row["n_seeds"] == 3 and row["n_censored"] == 0
