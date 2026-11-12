"""Phase 8b (CIR) — subspace-projected weight displacement (the CIR mechanism gate).

CIR claims durable unlearning *without* large global weight movement: it should displace
WITHIN the family-relevant (kept) subspace while sparing the shared (collapsed) subspace.
Global L2 (Phase 8) cannot see that. This probe splits ΔW = θ_B − θ₀ of each targeted
``output.dense.weight`` into kept vs collapsed components using the saved collapse bases,
and contrasts B with the compute-matched control A′ projected the same way.

Mechanism gate: B's KEPT-subspace displacement materially exceeds A′'s AND the
LMC barrier to base (Phase 8) is non-trivial. Failing both = dormant-not-deleted recurrence.

This is pure weight-space math (no model forward passes), so it is fast.

    python scripts/08b_cir_subspace.py model=esm2_35m

Reads checkpoints/{B,A_prime}, results/cir_subspace.npz (+ optional results/weightspace.json
for the LMC barrier); writes results/cir_weightspace.json.
"""
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 8b (CIR) — SUBSPACE-PROJECTED WEIGHT DISPLACEMENT")
    import numpy as np

    from src.analysis.weightspace import l2_by_group, model_state_dict_cpu, projected_l2_by_group
    from src.models.load import load_checkpoint
    from src.utils.io import load_json, save_json

    results = cfg.paths.results
    device = "cpu"  # weight math only; keep off-GPU

    npz_path = os.path.join(results, "cir_subspace.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found — run scripts/03b_train_cir.py first.")
    data = np.load(npz_path)
    layers = [int(x) for x in data["layers"]]
    bases = {
        li: {
            "act_basis": data[f"act_basis_{li}"],
            "grad_basis": data[f"grad_basis_{li}"],
        }
        for li in layers
    }

    sd0 = model_state_dict_cpu(load_checkpoint(cfg.model.name, device=device))
    sdB = model_state_dict_cpu(load_checkpoint(os.path.join(cfg.paths.checkpoints, "B"), device=device))
    proj_B = projected_l2_by_group(sd0, sdB, bases)
    glob_B = l2_by_group(sd0, sdB)

    record = {"layers": layers, "projected": {"B": proj_B}, "global": {"B": glob_B}}
    proj_Ap = None
    ap_path = os.path.join(cfg.paths.checkpoints, "A_prime")
    if os.path.exists(ap_path):
        sdAp = model_state_dict_cpu(load_checkpoint(ap_path, device=device))
        proj_Ap = projected_l2_by_group(sd0, sdAp, bases)
        record["projected"]["A_prime"] = proj_Ap
        record["global"]["A_prime"] = l2_by_group(sd0, sdAp)

    # fold in the LMC barrier from Phase 8 if available (the second half of the gate)
    lmc = None
    ws_path = os.path.join(results, "weightspace.json")
    if os.path.exists(ws_path):
        ws = load_json(ws_path)
        lmc = ws.get("lmc_B")
        if lmc is not None:
            record["lmc_B"] = {"f_barrier_nats": lmc.get("f_barrier_nats"),
                               "f_monotonic": lmc.get("f_monotonic")}
    save_json(record, os.path.join(results, "cir_weightspace.json"))

    # ---- report ----
    _bootstrap.banner("CIR MECHANISM GATE")
    keptB = proj_B["__total__"]["kept_l2"]
    collB = proj_B["__total__"]["collapsed_l2"]
    print(f"  B targeted ΔW:  kept={keptB:.4f}  collapsed={collB:.4f}  "
          f"(kept/collapsed = {keptB/(collB+1e-12):.2f})")
    ratio = float("nan")
    if proj_Ap is not None:
        keptAp = proj_Ap["__total__"]["kept_l2"]
        ratio = keptB / (keptAp + 1e-12)
        print(f"  A′ targeted ΔW: kept={keptAp:.4f}   ->  B/A′ kept-displacement = {ratio:.2f}×")
    mech_disp = (ratio > 1.5) if proj_Ap is not None else None
    if lmc is not None:
        barrier_ok = (not lmc.get("f_monotonic", True)) or (lmc.get("f_barrier_nats", 0.0) >= 0.5)
        print(f"  LMC θ₀→θ_B: barrier={lmc.get('f_barrier_nats')!r} nats  monotonic={lmc.get('f_monotonic')!r}")
        if mech_disp is not None:
            verdict = "PASS (kept displacement > A′ AND non-trivial barrier)" if (mech_disp and barrier_ok) \
                else "FAIL -> dormant-not-deleted recurrence"
            print(f"  => mechanism gate: {verdict}")
    else:
        print("  (run scripts/08_weightspace.py for the LMC barrier to complete the gate)")
    print(f"\nartifact: {os.path.join(results, 'cir_weightspace.json')}")


if __name__ == "__main__":
    main()
