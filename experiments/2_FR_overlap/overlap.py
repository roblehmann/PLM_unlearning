"""Score O(F,R) — the activation-subspace overlap between each candidate F and the
fixed retain set R — for every candidate pair, reusing the pipeline's own Phase 8e
logic (src.analysis.cir_fvsr.subspace_overlap, scripts/08e_cir_overlap.py) unmodified.

High overlap means little F-distinctive representational signal exists outside R's
subspace for a retain-preserving safeguard (e.g. CIR) to touch — this is a second,
independent axis (orthogonal to the PPL headroom test in ../1_naive_PPL_Floor) for
comparing candidate anchor families before committing to a full pipeline rerun.

    python experiments/2_FR_overlap/overlap.py

Reads candidate F_train fasta from experiments/0_F_candidate_PPL/data (not duplicated
here) and the canonical retain set at data/splits/retain_train.fasta (same R reused,
unchanged, across every prior run in this project). Writes results/overlap.csv
(one row per pair: aggregate O_topk/O_var/removable_topk/mean_cos_principal_angle)
and results/overlap_per_layer.json (full per-layer detail per pair).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES_DIR = ROOT / "experiments" / "0_F_candidate_PPL"

from src.analysis.cir_fvsr import subspace_overlap  # noqa: E402
from src.models.load import load_checkpoint, load_tokenizer  # noqa: E402
from src.train.cir import resolve_layers  # noqa: E402
from src.utils.io import ensure_dir, fasta_seqs, save_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(CANDIDATES_DIR / "data"), help="dir containing pairN/F_train.fasta")
    p.add_argument("--retain-fasta", default=str(ROOT / "data" / "splits" / "retain_train.fasta"),
                   help="fixed canonical R, same for every pair")
    p.add_argument("--out", default=str(HERE / "results" / "overlap.csv"))
    p.add_argument("--per-layer-out", default=str(HERE / "results" / "overlap_per_layer.json"))
    p.add_argument("--model", default="facebook/esm2_t12_35M_UR50D", help="HF id (esm2_35m headline model)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer-band", default="3,4,5,6,7,8", help="comma-separated layer indices; matches configs/blocking/cir.yaml")
    p.add_argument("--n-pcs", type=int, default=24)
    p.add_argument("--corpus-seqs", type=int, default=2048)
    p.add_argument("--pc-batch", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    pair_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and (d / "F_train.fasta").exists())
    if not pair_dirs:
        raise SystemExit(f"no pairN/F_train.fasta found under {data_dir}")

    print(f"Loading base model {args.model} on {args.device} ...")
    tok = load_tokenizer(args.model)
    base = load_checkpoint(args.model, device=args.device).eval()
    layer_band = [int(x) for x in args.layer_band.split(",") if x.strip() != ""]
    layers = resolve_layers(base, layer_band)
    print(f"targeted layers={layers}  n_pcs={args.n_pcs}  corpus_seqs={args.corpus_seqs}")

    retain_seqs = fasta_seqs(args.retain_fasta)
    print(f"R={len(retain_seqs)} retain seqs (fixed across all pairs)")

    rows = []
    per_layer_all = {}
    for pair_dir in pair_dirs:
        pair = pair_dir.name
        forget_seqs = fasta_seqs(pair_dir / "F_train.fasta")
        ov = subspace_overlap(
            base, tok, layers, retain_seqs, forget_seqs,
            n_pcs=args.n_pcs, corpus_seqs=args.corpus_seqs, pc_batch=args.pc_batch,
            max_length=args.max_length, seed=args.seed, device=args.device,
        )
        per_layer_all[pair] = ov["per_layer"]
        a = ov["aggregate"]
        print(f"  {pair:>6} F={len(forget_seqs):>6} -> O_topk={a['O_topk']:.3f} "
              f"O_var={a['O_var']:.3f} removable_topk={a['removable_topk']:.3f} "
              f"mean_cosθ={a['mean_cos_principal_angle']:.3f}")
        rows.append({
            "pair": pair, "n_layers": len(layers), "layer_band": ",".join(str(l) for l in layers),
            "n_pcs": ov["n_pcs"], "n_forget_available": len(forget_seqs),
            "O_topk": a["O_topk"], "O_var": a["O_var"],
            "removable_topk": a["removable_topk"],
            "mean_cos_principal_angle": a["mean_cos_principal_angle"],
        })

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("removable_topk", ascending=False)
    ensure_dir(Path(args.out).parent)
    df.to_csv(args.out, index=False)
    save_json(per_layer_all, args.per_layer_out)
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.per_layer_out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
