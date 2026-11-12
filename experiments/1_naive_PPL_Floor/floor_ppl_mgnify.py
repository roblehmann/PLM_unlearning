"""Score PPL_floor (base ESM-2 full-fine-tuned to convergence) for each MGnify F
candidate, combined with experiments/0_F_candidate_PPL/results/naive_ppl_mgnify.csv
into rel_headroom / tau / headroom_ok — the MGnify counterpart to floor_ppl.py,
reusing the same gate logic (src.eval.pseudo_perplexity.acquisition_threshold,
src.train.attack.run_attack) unmodified.

MGnify seed sets have no pre-built train/test split (unlike the Pfam pairs, which
came pre-split) and are small (67-1319 seqs per family) — so the split is done at
runtime via random_split, the SAME train/test splitter the real pipeline's
01_build_data.py uses (deterministic by --split-seed), rather than materializing new
fasta files on disk.

    python experiments/1_naive_PPL_Floor/floor_ppl_mgnify.py

Reads the F-role rows of experiments/0_F_candidate_PPL/data/mgnify_seed_seqs/manifest.tsv
and this experiment's own naive_ppl_mgnify.csv for PPL_naive. Writes
results/floor_ppl_mgnify.csv, plus live per-epoch history (same as floor_ppl.py) to
results/runs/<pair>.jsonl and results/history/<pair>.json.
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

from src.data.datamodule import random_split, subsample  # noqa: E402
from src.eval.pseudo_perplexity import acquisition_threshold  # noqa: E402
from src.models.load import load_tokenizer  # noqa: E402
from src.train.attack import run_attack  # noqa: E402
from src.utils.io import ensure_dir, read_fasta, save_json  # noqa: E402
from src.utils.tracking import JSONLTracker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(CANDIDATES_DIR / "data" / "mgnify_seed_seqs" / "manifest.tsv"))
    p.add_argument("--naive-csv", default=str(CANDIDATES_DIR / "results" / "naive_ppl_mgnify.csv"),
                   help="experiment-0 output, used for PPL_naive per pair's F family")
    p.add_argument("--out", default=str(HERE / "results" / "floor_ppl_mgnify.csv"))
    p.add_argument("--model", default="facebook/esm2_t12_35M_UR50D", help="HF id (esm2_35m headline model)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-frac", type=float, default=0.7, help="matches configs/data/pfam_default.yaml splits.forget_train_frac")
    p.add_argument("--split-seed", type=int, default=0)
    # mirrors cfg.floor.* / configs/attack/full_ft.yaml / configs/metrics/thresholds.yaml
    p.add_argument("--max-train-seqs", type=int, default=8192)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1.0e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--tau-rho", type=float, default=0.80)
    p.add_argument("--max-eval-seqs", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--masks-per-pass", type=int, default=8)
    # eval_every=1 by default (unlike floor_ppl.py's legacy default of 4): with these
    # small pools patience needs a real chance to fire, and scoring every epoch here is
    # cheap. Same reasoning as the widened 02_baseline_attack.py rerun this session.
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_ppl_naive(naive_csv: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(naive_csv)
    f_rows = df[df["role"] == "F"]
    return dict(zip(f_rows["pair"], f_rows["ppl_naive"]))


def main():
    args = parse_args()
    import pandas as pd

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest_dir = manifest_path.parent
    f_rows = manifest[manifest["role"] == "F"]

    ppl_naive_by_pair = load_ppl_naive(Path(args.naive_csv))

    tok = load_tokenizer(args.model)
    attack_cfg = {
        "mode": "full_ft", "lr": args.lr, "batch_size": args.batch_size,
        "mlm_probability": args.mlm_probability, "max_epochs": args.max_epochs,
        "weight_decay": 0.0, "patience": args.patience,
    }

    rows = []
    for _, r in f_rows.iterrows():
        pair = f"pair{int(r['pair_idx'])}"
        if pair not in ppl_naive_by_pair:
            print(f"[{pair}] skipped: no PPL_naive row in {args.naive_csv}")
            continue
        ppl_naive = float(ppl_naive_by_pair[pair])

        records = read_fasta(manifest_dir / str(r["fasta"]))
        train_recs, test_recs = random_split(records, args.train_frac, seed=args.split_seed)
        train_seqs = subsample([x.seq for x in train_recs], args.max_train_seqs, seed=args.seed)
        test_seqs = [x.seq for x in test_recs]
        print(f"[{pair}] {r['mgnifam_id']}: full-FT on {len(train_seqs)} train / "
              f"{len(test_seqs)} test seqs (of {len(records)} seed total, "
              f"train_frac={args.train_frac}, max_epochs={args.max_epochs}) ...")

        results_dir = str(Path(args.out).parent)
        tracker = JSONLTracker(run_name=pair, results_dir=results_dir)
        res = run_attack(
            args.model, tok, train_seqs, test_seqs, attack_cfg, args.device,
            seed=args.seed, max_length=args.max_length, background_seqs=None,
            ppl_batch_size=args.eval_batch_size, eval_masks_per_pass=args.masks_per_pass,
            max_eval_seqs=args.max_eval_seqs, eval_every=args.eval_every, tracker=tracker,
        )
        tracker.finish()
        save_json(res["history"], Path(results_dir) / "history" / f"{pair}.json")
        ppl_floor = res["best_target_ppl"]
        tau = acquisition_threshold(ppl_naive, ppl_floor, args.tau_rho)
        rel_headroom = (ppl_naive - ppl_floor) / ppl_naive if ppl_naive else 0.0
        headroom_ok = rel_headroom > 0.05 and ppl_naive > tau

        print(f"  -> PPL_naive={ppl_naive:.4f} PPL_floor={ppl_floor:.4f} "
              f"rel_headroom={rel_headroom:.3f} tau={tau:.4f} headroom_ok={headroom_ok} "
              f"(best_epoch={res['best_epoch']}/{res['epochs_run']})")

        rows.append({
            "pair": pair, "mgnifam_id": r["mgnifam_id"], "n_seed_total": len(records),
            "n_train_used": len(train_seqs), "n_test": len(test_seqs),
            "ppl_naive": ppl_naive, "ppl_floor": ppl_floor,
            "rel_headroom": rel_headroom, "tau": tau, "headroom_ok": headroom_ok,
            "epochs_run": res["epochs_run"], "best_epoch": res["best_epoch"],
        })

        # Flush after every pair (not just at the end) so a timeout mid-run doesn't
        # lose already-finished pairs — these full-length-protein fits are slow enough
        # that losing partial progress would be costly.
        ensure_dir(Path(args.out).parent)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"  (checkpointed {len(rows)}/{len(f_rows)} pairs to {args.out})")

    df = pd.DataFrame(rows).sort_values("rel_headroom", ascending=False)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
