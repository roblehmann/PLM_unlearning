"""Phase 8d (CIR) — attacker repair-delta localization. The mechanistic payoff figure.

Reproduce ONE representative relearning attack (full-FT on clean F) against a CIR B, save
the relearned model θ_attacked, then localize the *repair* Δ_attack = θ_attacked − θ_B in
the same two subspaces used for the defender's displacement:

  (1) CIR's OWN kept/collapsed basis          (results/<op>/cir_subspace.npz, as Phase 8b)
  (2) the independent F-distinctive vs R-shared activation basis (rebuilt on θ₀, Phase 8c)

plus a direct alignment of Δ_attack with the CIR update Δ_CIR = θ_B − θ₀.

Two clean hypotheses:
  H1  literal in-basin reversal — Δ_attack mostly NEGATES CIR's own update (high
      reversal_fraction, cos(Δ_attack, Δ_CIR) ≪ 0, repair concentrated in CIR's kept
      subspace). Consistent with the LMC finding.
  H2  entanglement / reconstruction — Δ_attack lands in the R_shared subspace CIR had to
      SPARE to hold utility. Then F is rebuilt from directions no utility-preserving
      safeguard is allowed to touch: the entanglement thesis made mechanistically concrete
      and the biosecurity-relevant conclusion.

Cost: one full-FT attack cell (reuses src.train.attack.run_attack unchanged except an
opt-in save) + one linear-algebra pass over checkpoints.

    python scripts/08d_cir_repair.py model=esm2_35m blocking=cir \
      +op=cir35m_peak +budget_n=4096

Writes results/<op>/cir_repair.json (+ printed report).
"""
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 8d (CIR) — ATTACKER REPAIR-DELTA LOCALIZATION")
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from src.analysis.cir_fvsr import build_fvsr_bases
    from src.analysis.weightspace import (fvsr_projected_l2, model_state_dict_cpu,
                                          projected_l2_by_group)
    from src.data.datamodule import DataPaths, attacker_trainset, target_test_seqs
    from src.models.load import load_checkpoint, load_tokenizer
    from src.train.attack import run_attack
    from src.train.cir import resolve_layers
    from src.utils.io import ensure_dir, fasta_seqs, save_json

    device = "cuda" if torch.cuda.is_available() else "cpu"
    op = str(cfg.get("op", "cir35m_peak"))
    budget_n = int(cfg.get("budget_n", 4096))
    attack_seed = int(cfg.get("attack_seed", 0))
    n_eval = int(cfg.get("n_eval", 256))
    n_pcs = int(cfg.get("fvsr_n_pcs", 24))
    n_corpus = int(cfg.get("fvsr_corpus_seqs", 2048))
    max_len = int(cfg.get("max_length", 1024))

    ck_root = cfg.paths.checkpoints
    rs_root = cfg.paths.results
    B_path = os.path.join(ck_root, op, "B")
    op_results = os.path.join(rs_root, op)
    attacked_path = os.path.join(ck_root, op, f"attacked_fullft_n{budget_n}")
    if not os.path.isdir(B_path):
        raise FileNotFoundError(f"{B_path} not found — expected a promoted CIR B for op '{op}'.")

    # attack recipe = the sweep's full_ft cell (absolute path, cwd-independent)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attack_cfg = OmegaConf.load(os.path.join(root, "configs", "attack", "full_ft.yaml"))
    if "attack_epochs" in cfg:
        attack_cfg.max_epochs = int(cfg.attack_epochs)
    assert str(attack_cfg.mode) == "full_ft", "repair-delta needs a full-FT attack (whole-weight delta)"

    tok = load_tokenizer(cfg.model.name)
    paths = DataPaths(cfg.paths.data)

    # ---- 1) reproduce one clean-F full-FT relearn on B; save θ_attacked ----
    seqs = attacker_trainset(paths, "F", "clean", budget_n, seed=attack_seed)
    f_test = target_test_seqs(paths, "F")
    print(f"op={op}  attack: full_ft clean F  budget_n={budget_n}  "
          f"epochs={int(attack_cfg.max_epochs)}  -> saving θ_attacked to {attacked_path}")
    atk = run_attack(
        checkpoint_path=B_path, tokenizer=tok, attacker_seqs=seqs, target_test_seqs=f_test,
        attack_cfg=attack_cfg, device=device, seed=attack_seed, max_length=max_len,
        ppl_batch_size=int(cfg.get("eval_batch_size", 64)),
        eval_masks_per_pass=int(cfg.get("eval_masks_per_pass", 8)),
        max_eval_seqs=n_eval, eval_every=int(cfg.get("eval_every", 1)),
        save_model_path=attacked_path,
    )
    print(f"  F_test pseudo-PPL:  naive(B) {atk['naive_target_ppl']:.3f} -> "
          f"best {atk['best_target_ppl']:.3f} (epoch {atk['best_epoch']})  "
          f"[base θ₀ F-PPL ≈ 7.74]  -> {'RECONSTRUCTED' if atk['best_target_ppl'] < 9.0 else 'partial'}")

    # ---- 2) referee basis (rebuilt on θ₀, same construction as Phase 8c) ----
    base = load_checkpoint(cfg.model.name, device=device).eval()
    layer_band = list(cfg.blocking.layer_band) if "blocking" in cfg else [3, 4, 5, 6, 7, 8]
    layers = resolve_layers(base, layer_band)
    fvsr_bases = build_fvsr_bases(
        base, tok, layers, fasta_seqs(paths.retain_train), fasta_seqs(paths.F_train),
        n_pcs=n_pcs, corpus_seqs=n_corpus, max_length=max_len, seed=cfg.seed, device=device)

    # CIR's own collapse basis (from the defender run)
    cir_npz = os.path.join(op_results, "cir_subspace.npz")
    cir_bases = None
    if os.path.exists(cir_npz):
        d = np.load(cir_npz)
        cir_bases = {int(li): {"act_basis": d[f"act_basis_{li}"], "grad_basis": d[f"grad_basis_{li}"]}
                     for li in d["layers"]}
    else:
        print(f"  (note: {cir_npz} missing — skipping CIR kept/collapsed decomposition)")

    # ---- 3) checkpoints -> deltas ----
    sd0 = model_state_dict_cpu(base.to("cpu"))
    sdB = model_state_dict_cpu(load_checkpoint(B_path, device="cpu"))
    sdX = model_state_dict_cpu(load_checkpoint(attacked_path, device="cpu"))

    tgt_keys = [f"esm.encoder.layer.{li}.output.dense.weight" for li in layers]

    def align(sd_a0, sd_a1, sd_b0, sd_b1, keys=None):
        """cos and reversal_fraction between Δa = a1−a0 and Δb = b1−b0 over `keys`
        (or all shared float tensors). reversal_fraction = −<Δa,Δb>/‖Δb‖²: how much of
        Δb (the CIR update) the repair Δa undoes along Δb's own direction."""
        dot = na = nb = 0.0
        for k, p in sd_a0.items():
            if keys is not None and k not in keys:
                continue
            if k not in sd_a1 or k not in sd_b0 or k not in sd_b1 or not torch.is_floating_point(p):
                continue
            da = (sd_a1[k].float() - sd_a0[k].float()).flatten()
            db = (sd_b1[k].float() - sd_b0[k].float()).flatten()
            dot += float((da * db).sum()); na += float((da * da).sum()); nb += float((db * db).sum())
        cos = dot / ((na ** 0.5) * (nb ** 0.5) + 1e-12)
        return {"cos": cos, "reversal_fraction": -dot / (nb + 1e-12),
                "attack_l2": na ** 0.5, "cir_l2": nb ** 0.5}

    record = {
        "op": op, "budget_n": budget_n, "layers": layers,
        "attack": {k: atk[k] for k in ("naive_target_ppl", "best_target_ppl", "final_target_ppl", "best_epoch", "epochs_run")},
        "alignment_vs_cir_update": {
            "targeted_dense": align(sdB, sdX, sd0, sdB, keys=set(tgt_keys)),
            "global": align(sdB, sdX, sd0, sdB, keys=None),
        },
        "repair_delta": {"fvsr": fvsr_projected_l2(sdB, sdX, fvsr_bases)},
        "cir_update": {"fvsr": fvsr_projected_l2(sd0, sdB, fvsr_bases)},
    }
    if cir_bases is not None:
        record["repair_delta"]["kept_collapsed"] = projected_l2_by_group(sdB, sdX, cir_bases)
        record["cir_update"]["kept_collapsed"] = projected_l2_by_group(sd0, sdB, cir_bases)

    ensure_dir(op_results)
    out_path = os.path.join(op_results, "cir_repair.json")
    save_json(record, out_path)

    # ---- report ----
    _bootstrap.banner("REPAIR-DELTA LOCALIZATION")
    a = record["alignment_vs_cir_update"]
    print("alignment  Δ_attack vs Δ_CIR (θ_B−θ₀):")
    for scope in ("targeted_dense", "global"):
        s = a[scope]
        print(f"  {scope:<15} cos={s['cos']:+.3f}  reversal_fraction={s['reversal_fraction']:+.3f}  "
              f"(‖Δ_attack‖={s['attack_l2']:.3f}  ‖Δ_CIR‖={s['cir_l2']:.3f})")

    def show(name, block):
        f = block["fvsr"]["__total__"]
        tot = f["total_l2"] + 1e-12
        line = (f"  {name:<12} F_distinct={f['f_distinctive_l2']:.3f} ({f['f_distinctive_l2']/tot:.2f})  "
                f"R_shared={f['r_shared_l2']:.3f} ({f['r_shared_l2']/tot:.2f})  "
                f"rest={f['rest_l2']:.3f}  total={f['total_l2']:.3f}")
        if "kept_collapsed" in block:
            kc = block["kept_collapsed"]["__total__"]
            line += f"   | CIR-basis kept={kc['kept_l2']:.3f} collapsed={kc['collapsed_l2']:.3f}"
        print(line)

    print("\nsubspace localization (F-distinctive / R-shared over targeted ΔW):")
    show("Δ_CIR", record["cir_update"])
    show("Δ_attack", record["repair_delta"])
    print(f"\nartifact: {out_path}")
    print("H1 (in-basin reversal): reversal_fraction→1, cos≪0, repair in CIR's kept subspace.")
    print("H2 (entanglement):      Δ_attack R_shared fraction high — F rebuilt via directions CIR spared.")


if __name__ == "__main__":
    main()
