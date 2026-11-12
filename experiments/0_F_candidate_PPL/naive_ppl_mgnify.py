"""Score PPL_naive (base ESM-2, no fine-tuning) on the MGnify metagenomic-family
candidates listed in data/mgnify_seed_seqs/manifest.tsv — the same naive-PPL screening
as naive_ppl.py, but for MGnify protein families (metagenomic "dark matter" clusters,
no Pfam/UniRef characterization) instead of the Pfam pair0-4 candidates. Reuses the
pipeline's own scoring code (src.eval.pseudo_perplexity.pseudo_perplexity) unmodified.

    python experiments/0_F_candidate_PPL/naive_ppl_mgnify.py

Each manifest row is a family's seed-sequence fasta (no train/test split provided —
these are the seed sets themselves), scored directly. Writes results/naive_ppl_mgnify.csv
(pair, role, mgnifam_id, fid, n_seed_total, n_scored, ppl_naive).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datamodule import subsample  # noqa: E402
from src.eval.pseudo_perplexity import pseudo_perplexity  # noqa: E402
from src.models.load import load_checkpoint, load_tokenizer  # noqa: E402
from src.utils.io import ensure_dir, fasta_seqs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(HERE / "data" / "mgnify_seed_seqs" / "manifest.tsv"))
    p.add_argument("--out", default=str(HERE / "results" / "naive_ppl_mgnify.csv"))
    p.add_argument("--model", default="facebook/esm2_t12_35M_UR50D", help="HF id (esm2_35m headline model)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-eval-seqs", type=int, default=256, help="matches cfg.eval.max_eval_seqs")
    p.add_argument("--batch-size", type=int, default=64, help="matches cfg.eval.batch_size")
    p.add_argument("--masks-per-pass", type=int, default=8, help="matches cfg.eval.masks_per_pass")
    p.add_argument("--max-length", type=int, default=1024, help="same truncation length used pipeline-wide")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    import pandas as pd

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest_dir = manifest_path.parent

    print(f"Loading base model {args.model} on {args.device} ...")
    model = load_checkpoint(args.model, device=args.device)
    tok = load_tokenizer(args.model)

    rows = []
    for _, r in manifest.iterrows():
        fasta = manifest_dir / str(r["fasta"])
        seqs = fasta_seqs(fasta)
        n_total = len(seqs)
        scored = subsample(seqs, args.max_eval_seqs, seed=args.seed)
        res = pseudo_perplexity(
            model, tok, scored, args.device,
            batch_size=args.batch_size, max_length=args.max_length,
            masks_per_pass=args.masks_per_pass,
        )
        pair = f"pair{int(r['pair_idx'])}"
        print(f"  {pair:>6}/{r['role']:<7} {r['mgnifam_id']} n={n_total:>5} scored={len(scored):>4} "
              f"-> PPL_naive={res.ppl:.4f}")
        rows.append({
            "pair": pair, "role": r["role"], "mgnifam_id": r["mgnifam_id"], "fid": int(r["fid"]),
            "fasta": str(fasta), "n_seed_total": n_total, "n_scored": len(scored), "ppl_naive": res.ppl,
        })

    df = pd.DataFrame(rows).sort_values(["pair", "role"])
    ensure_dir(Path(args.out).parent)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
