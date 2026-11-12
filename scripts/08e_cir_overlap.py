"""Phase 8e (CIR) — O(F,R) activation-subspace overlap. The upstream measurement behind
the CIR result, and the tie from that result back to the representation-mode question.

Principal angles between F's and R's top-k activation PCAs (on the frozen base θ₀) quantify
how much of the forget family's representational variance already lives inside the retain
subspace. High overlap means there is little family-distinctive signal to remove that a
utility-preserving safeguard is *allowed* to touch — so CIR (which must spare R) has little
room to act — and it predicts the shared-subspace attacker repair (Phase 8d, H2).

    python scripts/08e_cir_overlap.py model=esm2_35m blocking=cir

Reports O_topk / O_var / removable_topk and the principal-angle cosine spectrum per layer;
writes results/cir_overlap.json. One GPU pass over R + F on θ₀ (no attack, no checkpoints).
"""
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 8e (CIR) — O(F,R) SUBSPACE OVERLAP")
    import torch

    from src.analysis.cir_fvsr import subspace_overlap
    from src.data.datamodule import DataPaths
    from src.models.load import load_checkpoint, load_tokenizer
    from src.train.cir import resolve_layers
    from src.utils.io import ensure_dir, fasta_seqs, save_json

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_pcs = int(cfg.get("fvsr_n_pcs", 24))
    n_corpus = int(cfg.get("fvsr_corpus_seqs", 2048))
    max_len = int(cfg.get("max_length", 1024))

    tok = load_tokenizer(cfg.model.name)
    base = load_checkpoint(cfg.model.name, device=device).eval()
    layer_band = list(cfg.blocking.layer_band) if "blocking" in cfg else [3, 4, 5, 6, 7, 8]
    layers = resolve_layers(base, layer_band)
    paths = DataPaths(cfg.paths.data)
    print(f"targeted layers={layers}  n_pcs={n_pcs}  corpus_seqs={n_corpus}  device={device}")

    ov = subspace_overlap(
        base, tok, layers, fasta_seqs(paths.retain_train), fasta_seqs(paths.F_train),
        n_pcs=n_pcs, corpus_seqs=n_corpus, max_length=max_len, seed=cfg.seed, device=device)

    ensure_dir(cfg.paths.results)
    out_path = os.path.join(cfg.paths.results, "cir_overlap.json")
    save_json(ov, out_path)

    _bootstrap.banner(f"O(F,R) — F variance inside R's top-{n_pcs} subspace")
    print(f"{'layer':<8}{'O_topk':>9}{'O_var':>9}{'removable':>11}{'#cosθ≥.9':>10}   cos θ (top angles)")
    for li in layers:
        d = ov["per_layer"][li]
        cs = d["cos_principal_angles"]
        head = " ".join(f"{c:.2f}" for c in cs[:8])
        print(f"{li:<8}{d['O_topk']:>9.3f}{d['O_var']:>9.3f}{d['removable_topk']:>11.3f}"
              f"{d['n_angles_ge_0.9']:>10}   {head}")
    a = ov["aggregate"]
    print(f"\nAGGREGATE  O_topk={a['O_topk']:.3f}  O_var={a['O_var']:.3f}  "
          f"removable_topk={a['removable_topk']:.3f}  mean cosθ={a['mean_cos_principal_angle']:.3f}")
    print(f"\nartifact: {out_path}")
    print("high O_topk / low removable_topk => little family signal exists outside R's subspace: "
          "CIR (which must spare R to hold utility) cannot delete durably, and the repair returns "
          "through the shared subspace (Phase 8d H2).")


if __name__ == "__main__":
    main()
