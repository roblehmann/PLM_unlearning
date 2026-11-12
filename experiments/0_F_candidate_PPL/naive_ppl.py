"""Score PPL_naive (base ESM-2, no fine-tuning) on every candidate test set under
data/pairN/*_test.fasta — the same base-model scoring step used as the first half
of the headroom gate in scripts/02_baseline_attack.py, reused here unmodified so a
new anchor family can be screened before committing a pair to a full pipeline run.

    python experiments/0_F_candidate_PPL/naive_ppl.py
    python experiments/0_F_candidate_PPL/naive_ppl.py --model facebook/esm2_t12_35M_UR50D --device cuda

Writes results/naive_ppl.csv (pair, family, fasta, n_seqs, n_scored, ppl_naive).
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
    p.add_argument("--data-dir", default=str(HERE / "data"), help="dir containing pairN/*_test.fasta")
    p.add_argument("--out", default=str(HERE / "results" / "naive_ppl.csv"))
    p.add_argument("--model", default="facebook/esm2_t12_35M_UR50D", help="HF id (esm2_35m headline model)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-eval-seqs", type=int, default=256, help="matches cfg.eval.max_eval_seqs")
    p.add_argument("--batch-size", type=int, default=64, help="matches cfg.eval.batch_size")
    p.add_argument("--masks-per-pass", type=int, default=8, help="matches cfg.eval.masks_per_pass")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    pair_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not pair_dirs:
        raise SystemExit(f"no pair dirs found under {data_dir}")

    targets = []  # (pair, family_label, fasta_path)
    for pair_dir in pair_dirs:
        for fasta in sorted(pair_dir.glob("*_test.fasta")):
            label = fasta.stem[: -len("_test")] if fasta.stem.endswith("_test") else fasta.stem
            targets.append((pair_dir.name, label, fasta))

    print(f"Loading base model {args.model} on {args.device} ...")
    model = load_checkpoint(args.model, device=args.device)
    tok = load_tokenizer(args.model)

    rows = []
    for pair, label, fasta in targets:
        seqs = fasta_seqs(fasta)
        n_total = len(seqs)
        scored = subsample(seqs, args.max_eval_seqs, seed=args.seed)
        res = pseudo_perplexity(
            model, tok, scored, args.device,
            batch_size=args.batch_size, max_length=args.max_length,
            masks_per_pass=args.masks_per_pass,
        )
        print(f"  {pair:>6}/{label:<12} n={n_total:>6} scored={len(scored):>4} -> PPL_naive={res.ppl:.4f}")
        rows.append({
            "pair": pair, "family": label, "fasta": str(fasta),
            "n_seqs": n_total, "n_scored": len(scored), "ppl_naive": res.ppl,
        })

    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["pair", "family"])
    ensure_dir(Path(args.out).parent)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
