"""Sibling-ladder relearning-sweep orchestration (Phase 10).

A sibling of ``src/sweep/run_sweep.py``. Where the Phase-6 sweep varies
(adversary_data × target_family) over the fixed F / F' families, this sweep instead
varies over the **homology-graded ladder rungs** {F'₀, F'₁, …} of same-clan siblings
at increasing distance from F. Each cell fine-tunes a checkpoint on ONE rung's
training corpus (never F) and scores the held-out FORGET-family pseudo-perplexity
(F-PPL) — the inverse of the Phase-1 leakage-curve framing (tune toward F, read out
siblings). The question is whether tuning on a near neighbour of F reawakens the
blocked F capability, and at which rung it stops.

``build_ladder_cells`` enumerates the grid (pure; mirrors ``build_cells``).
``run_ladder_sweep`` executes the cells against the {B, A_prime} checkpoints, drawing
the attacker budget from ``data/ladder/<F>/corpora/<rung_acc>_ft.fasta`` (the rung's
full family with every F_test-matching sequence already removed) and scoring F-PPL on
the canonical ``data/splits/F_test.fasta``. Writes a tidy long-format CSV, one row per
model × recipe × rank × budget × seed × rung. Resumable / SLURM-array-shardable in the
same way as ``run_sweep`` (skip any cell already present in ANY shard CSV).
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path
from typing import Dict, List, Optional

from src.data.datamodule import DataPaths, subsample, target_test_seqs
from src.train.mlm import cfg_get
from src.utils.io import fasta_seqs, load_json

FIELDNAMES = [
    "model", "recipe", "lora_rank", "budget_n", "seed",
    "rung", "rung_acc", "rung_id", "mean_id_all", "hmm_bits_mean", "band",
    "best_target_ppl", "final_target_ppl", "naive_target_ppl", "best_epoch", "epochs_run",
    "membership_auroc", "n_train", "trainable_params", "flops_total", "wall_clock_s",
]


def load_rungs(ladder_dir: str, only: Optional[List[str]] = None) -> List[dict]:
    """Read the selected ladder rungs (accession + distance metadata) from
    ``<ladder_dir>/ladder_selected.json``. If ``only`` is given, keep just those
    accessions (preserving their ladder order)."""
    sel = load_json(Path(ladder_dir) / "ladder_selected.json")
    rungs = list(sel["rungs"])
    if only:
        want = set(only)
        rungs = [r for r in rungs if r["acc"] in want]
        missing = want - {r["acc"] for r in rungs}
        if missing:
            raise KeyError(f"rungs {sorted(missing)} not found in {ladder_dir}/ladder_selected.json")
    return rungs


def rung_corpus_path(ladder_dir: str, rung_acc: str) -> Path:
    """The rung's fine-tuning corpus: the full family minus every sequence matching
    F_test (built into data/ladder/<F>/corpora/<acc>_ft.fasta)."""
    return Path(ladder_dir) / "corpora" / f"{rung_acc}_ft.fasta"


def build_ladder_cells(sweep_cfg, rungs: List[dict]) -> List[dict]:
    """Cartesian product of the sweep axes × ladder rungs. LoRA varies over
    lora_ranks; full_ft has a single (rank=None) cell. Mirrors ``build_cells``."""
    models = list(cfg_get(sweep_cfg, "models", ["B", "A_prime"]))
    recipes = list(cfg_get(sweep_cfg, "recipes", ["lora", "full_ft"]))
    budgets = list(cfg_get(sweep_cfg, "budgets_n", [16, 64, 256, 1024, 4096]))
    seeds = list(cfg_get(sweep_cfg, "seeds", [0, 1, 2]))
    lora_ranks = list(cfg_get(sweep_cfg, "lora_ranks", [8, 32]))

    cells: List[dict] = []
    for recipe in recipes:
        ranks = lora_ranks if recipe == "lora" else [None]
        for model in models:
            for budget in budgets:
                for seed in seeds:
                    for rung in rungs:
                        for rank in ranks:
                            cells.append({
                                "model": model, "recipe": recipe, "lora_rank": rank,
                                "budget_n": budget, "seed": seed,
                                "rung": rung["rung"], "rung_acc": rung["acc"],
                                "rung_id": rung["id"], "mean_id_all": rung["mean_id_all"],
                                "hmm_bits_mean": rung["hmm_bits_mean"], "band": rung["band"],
                            })
    return cells


def _cell_key(c: dict) -> tuple:
    return (c["model"], c["recipe"], c["lora_rank"], c["budget_n"], c["seed"], c["rung_acc"])


def _load_done(out_csv: Path) -> set:
    done = set()
    if out_csv.exists():
        with open(out_csv) as fh:
            for row in csv.DictReader(fh):
                done.add((
                    row["model"], row["recipe"],
                    None if row["lora_rank"] in ("", "None") else int(row["lora_rank"]),
                    int(row["budget_n"]), int(row["seed"]), row["rung_acc"],
                ))
    return done


def run_ladder_sweep(
    cfg,
    ladder_dir: str,
    model_paths: Dict[str, str],
    out_csv: str,
    attack_cfgs: Optional[Dict[str, object]] = None,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = True,
    tracker=None,
) -> str:
    """Execute the ladder sweep (or a shard of it). ``model_paths`` maps 'B'/'A_prime'
    to a checkpoint dir. ``attack_cfgs`` maps each recipe ('lora','full_ft') to its
    config node. Returns the CSV path written."""
    from src.models.load import load_tokenizer
    from src.train.attack import run_attack

    only = cfg_get(cfg.sweep, "rungs", None)
    rungs = load_rungs(ladder_dir, list(only) if only else None)
    cells = build_ladder_cells(cfg.sweep, rungs)
    if num_shards > 1:
        cells = cells[shard_index::num_shards]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Global resume: skip any cell already completed in ANY shard file.
    done = set()
    if resume:
        for p in glob.glob(str(out_csv.parent / "*.csv")):
            done |= _load_done(Path(p))
    write_header = not out_csv.exists()

    tokenizer = load_tokenizer(cfg.model.name)
    device = cfg.device
    max_length = int(cfg.data.max_length)
    eval_cfg = cfg.eval if hasattr(cfg, "eval") else {}
    n_eval = int(cfg_get(eval_cfg, "max_eval_seqs", 256))
    eval_bs = int(cfg_get(eval_cfg, "batch_size", 64))
    eval_mpp = int(cfg_get(eval_cfg, "masks_per_pass", 8))
    eval_ev = int(cfg_get(eval_cfg, "eval_every", 4))

    # Held-out F-PPL is scored on the canonical F_test for EVERY cell (loaded once).
    paths = DataPaths(cfg.paths.data)
    f_test = target_test_seqs(paths, "F")

    background = []
    if Path(paths.background).exists():
        background = fasta_seqs(paths.background)

    # Per-rung corpora are large; cache the loaded sequence list so we read each fasta
    # from disk at most once per shard (budget subsampling is done per cell downstream).
    corpus_cache: Dict[str, List[str]] = {}

    def _corpus(acc: str) -> List[str]:
        if acc not in corpus_cache:
            corpus_cache[acc] = fasta_seqs(rung_corpus_path(ladder_dir, acc))
        return corpus_cache[acc]

    if attack_cfgs is None:
        attack_cfgs = {cfg.attack.mode: cfg.attack}

    with open(out_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            fh.flush()
        for c in cells:
            if _cell_key(c) in done:
                continue
            if c["recipe"] not in attack_cfgs:
                raise KeyError(f"No attack config provided for recipe '{c['recipe']}'. "
                               f"Pass attack_cfgs={{'lora':..., 'full_ft':...}}.")
            attack_cfg = _attack_cfg_for(attack_cfgs[c["recipe"]], c["recipe"], c["lora_rank"])
            # budget_n sibling examples drawn from the rung corpus (deterministic by seed).
            seqs = subsample(_corpus(c["rung_acc"]), c["budget_n"], seed=c["seed"])
            res = run_attack(
                checkpoint_path=model_paths[c["model"]],
                tokenizer=tokenizer,
                attacker_seqs=seqs,
                target_test_seqs=f_test,
                attack_cfg=attack_cfg,
                device=device,
                seed=c["seed"],
                max_length=max_length,
                background_seqs=background,
                ppl_batch_size=eval_bs,
                eval_masks_per_pass=eval_mpp,
                max_eval_seqs=n_eval,
                eval_every=eval_ev,
            )
            row = {
                **c,
                "best_target_ppl": res["best_target_ppl"],
                "final_target_ppl": res["final_target_ppl"],
                "naive_target_ppl": res["naive_target_ppl"],
                "best_epoch": res["best_epoch"],
                "epochs_run": res["epochs_run"],
                "membership_auroc": res["membership_auroc"],
                "n_train": res["n_train"],
                "trainable_params": res["trainable_params"],
                "flops_total": res["flops"]["total_flops"],
                "wall_clock_s": res["flops"]["wall_clock_s"],
            }
            writer.writerow(row)
            fh.flush()
            if tracker:
                tracker.log({k: row[k] for k in ("best_target_ppl", "budget_n", "model", "rung_acc")})
            print(f"[ladder] {_cell_key(c)} -> best_ppl={res['best_target_ppl']:.4f}")
    return str(out_csv)


def _attack_cfg_for(attack_cfg, recipe: str, lora_rank):
    """Clone a recipe's attack config node and override mode/rank for this cell.
    (Same helper as run_sweep._attack_cfg_for.)"""
    from omegaconf import OmegaConf

    node = OmegaConf.create(OmegaConf.to_container(attack_cfg, resolve=True))
    node.mode = recipe
    if recipe == "lora" and lora_rank is not None:
        node.lora_rank = int(lora_rank)
    return node
