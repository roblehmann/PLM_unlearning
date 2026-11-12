"""Phase 8c (CIR) — F-distinctive-vs-R-shared subspace-projected displacement.

The airtight version of the CIR falsification. Phase 8 (global L2) is dismissible for a
method whose whole claim is *surgical movement in the right subspace*, and Phase 8b's
kept/collapsed split uses CIR's OWN collapse basis (advocate can call it circular). This
probe builds an **independent referee's basis** — activation PCA over R vs over F on the
frozen base θ₀ — and asks, for every operating point, how ‖θ_B − θ₀‖ divides among:

  * F_distinctive : directions the forget family uses that general proteins do not
                    (span of F's activation PCs, orthogonalised against R's),
  * R_shared      : the general/shared activation directions (R's activation PCs),
  * rest          : low-variance-for-both directions.

Read against the durability multiplier (Phase 6/7), this forces the dilemma the advocate
cannot escape: peak either (a) barely moved in F_distinctive — so its "surgical" deletion
did not even land in the family-specific subspace — or (b) moved substantially there and
was STILL reversed by the attacker (M≈1×) — so surgical movement in the right subspace is
not sufficient for durability. Either branch falsifies the CIR robustness claim.

One GPU pass on θ₀ builds the R/F bases; the ΔW decomposition for each B (and A′) is pure
weight math. Point it at a comma-separated list of operating-point result dirs.

    python scripts/08c_cir_fvsr.py model=esm2_35m \
      +ops=cir35m_lowdis,cir35m_peak,cir_forgetpca/main

Writes results/cir_fvsr.json (+ a printed table). Base R/F sequences come from the same
CL0063 DataPaths every other phase uses.
"""
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 8c (CIR) — F-DISTINCTIVE vs R-SHARED DISPLACEMENT")
    import numpy as np
    import torch

    from src.analysis.cir_fvsr import build_fvsr_bases
    from src.analysis.weightspace import fvsr_projected_l2, model_state_dict_cpu
    from src.data.datamodule import DataPaths
    from src.models.load import load_checkpoint, load_tokenizer
    from src.train.cir import resolve_layers
    from src.utils.io import ensure_dir, fasta_seqs, save_json

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = cfg.paths.results
    n_pcs = int(cfg.get("fvsr_n_pcs", 24))          # PCs per corpus (parity with CIR n_act_pcs)
    n_corpus = int(cfg.get("fvsr_corpus_seqs", 2048))
    pc_batch = int(cfg.get("fvsr_batch", 16))
    max_len = int(cfg.get("max_length", 1024))
    tol = 1e-4                                       # drop F directions fully inside R's span

    # operating points to decompose: result subdirs each holding checkpoints/<dir>/B.
    # Hydra turns `+ops=a,b,c` into a list; also accept a plain comma string.
    ops_raw = cfg.get("ops", "cir35m_lowdis,cir35m_peak,cir_forgetpca/main")
    if isinstance(ops_raw, str):
        ops = [o.strip() for o in ops_raw.split(",")]
    else:
        ops = [str(o).strip() for o in ops_raw]
    ops = [o for o in ops if o]

    tok = load_tokenizer(cfg.model.name)
    base = load_checkpoint(cfg.model.name, device=device).eval()
    layer_band = list(cfg.blocking.layer_band) if "blocking" in cfg else [3, 4, 5, 6, 7, 8]
    layers = resolve_layers(base, layer_band)
    print(f"targeted layers={layers}  n_pcs={n_pcs}  corpus_seqs={n_corpus}  device={device}")

    paths = DataPaths(cfg.paths.data)
    R = fasta_seqs(paths.retain_train)
    F = fasta_seqs(paths.F_train)
    print(f"R={len(R)} retain seqs   F={len(F)} forget seqs  -> building activation PCA bases on θ₀")
    fvsr_bases = build_fvsr_bases(
        base, tok, layers, R, F,
        n_pcs=n_pcs, corpus_seqs=n_corpus, pc_batch=pc_batch, max_length=max_len,
        seed=cfg.seed, device=device, tol=tol,
    )
    dims = {li: (int(fvsr_bases[li]["F_distinctive"].shape[0]), int(fvsr_bases[li]["R_shared"].shape[0]))
            for li in layers}
    print("per-layer (dim F_distinctive, dim R_shared):", dims)

    # persist the bases so Phase 8d (attacker repair-delta) reuses the SAME referee subspace
    ensure_dir(results)
    bpack = {"layers": np.array(layers)}
    for li in layers:
        bpack[f"F_distinctive_{li}"] = fvsr_bases[li]["F_distinctive"]
        bpack[f"R_shared_{li}"] = fvsr_bases[li]["R_shared"]
    np.savez(os.path.join(results, "cir_fvsr_bases.npz"), **bpack)

    # ---- decompose ΔW for every operating point (B and, if present, A′) ----
    sd0 = model_state_dict_cpu(base.to("cpu"))
    record = {"layers": layers, "n_pcs": n_pcs, "dims": {str(k): v for k, v in dims.items()},
              "ops": {}}
    rows = []
    for op in ops:
        entry = {}
        for arm in ("B", "A_prime"):
            ck = os.path.join(cfg.paths.checkpoints, op, arm)
            if not os.path.isdir(ck):
                continue
            sd = model_state_dict_cpu(load_checkpoint(ck, device="cpu"))
            proj = fvsr_projected_l2(sd0, sd, fvsr_bases)
            entry[arm] = proj
            t = proj["__total__"]
            rows.append((op, arm, t["f_distinctive_l2"], t["r_shared_l2"], t["rest_l2"], t["total_l2"]))
        record["ops"][op] = entry

    ensure_dir(results)
    out_path = os.path.join(results, "cir_fvsr.json")
    save_json(record, out_path)

    _bootstrap.banner("F-DISTINCTIVE vs R-SHARED DISPLACEMENT (‖ΔW‖ over targeted layers)")
    print(f"{'operating point':<24}{'arm':<9}{'F_distinct':>12}{'R_shared':>11}{'rest':>9}"
          f"{'total':>9}{'F/total':>9}")
    for op, arm, f, r, rest, tot in rows:
        frac = f / (tot + 1e-12)
        print(f"{op:<24}{arm:<9}{f:>12.4f}{r:>11.4f}{rest:>9.4f}{tot:>9.4f}{frac:>9.3f}")
    print(f"\nartifact: {out_path}")
    print("read F/total against the durability multiplier: low F/total => never moved in the "
          "family subspace; high F/total + M≈1× => moved there and was still reversed.")


if __name__ == "__main__":
    main()
