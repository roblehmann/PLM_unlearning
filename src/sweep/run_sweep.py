"""Attacker-sweep orchestration (Phase 6).

`build_cells` enumerates the grid (pure; unit-tested). `run_sweep` executes the
cells against the {A, B, A'} checkpoints and writes a tidy long-format table — one
row per model × recipe × rank × budget × seed × adversary_data × target_family.

Designed for SLURM array jobs: pass (shard_index, num_shards) to run a slice, and
each shard appends to its own CSV (concatenated at analysis time). Rows are flushed
as they complete and existing rows are skipped, so jobs are resumable/idempotent.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path
from typing import Dict, List, Optional

from src.data.datamodule import DataPaths, attacker_trainset, target_test_seqs
from src.train.mlm import cfg_get

FIELDNAMES = [
    "model", "recipe", "lora_rank", "budget_n", "seed", "adversary_data", "target_family",
    "best_target_ppl", "final_target_ppl", "naive_target_ppl", "best_epoch", "epochs_run",
    "membership_auroc", "n_train", "trainable_params", "flops_total", "wall_clock_s",
]


def build_cells(sweep_cfg) -> List[dict]:
    """Cartesian product of the sweep axes. LoRA varies over lora_ranks; full_ft has
    a single (rank=None) cell.

    ``adversary_data`` may be 'clean', 'homolog_augmented', or 'retain_only'. For
    'retain_only' (the relearning probe) the attacker trains on the retain set R for
    *every* target_family — only the held-out family whose PPL is scored differs — so
    (retain_only, F) and (retain_only, F_prime) are the same training run scored
    against two targets. See ``datamodule.attacker_trainset``."""
    models = list(cfg_get(sweep_cfg, "models", ["A", "B", "A_prime"]))
    recipes = list(cfg_get(sweep_cfg, "recipes", ["lora", "full_ft"]))
    budgets = list(cfg_get(sweep_cfg, "budgets_n", [16, 64, 256, 1024, 4096]))
    seeds = list(cfg_get(sweep_cfg, "seeds", [0, 1, 2]))
    lora_ranks = list(cfg_get(sweep_cfg, "lora_ranks", [8, 32]))
    adversary_data = list(cfg_get(sweep_cfg, "adversary_data", ["clean", "homolog_augmented"]))
    target_family = list(cfg_get(sweep_cfg, "target_family", ["F", "F_prime"]))

    cells: List[dict] = []
    for recipe in recipes:
        ranks = lora_ranks if recipe == "lora" else [None]
        for model in models:
            for budget in budgets:
                for seed in seeds:
                    for adv in adversary_data:
                        for tgt in target_family:
                            for rank in ranks:
                                cells.append({
                                    "model": model, "recipe": recipe, "lora_rank": rank,
                                    "budget_n": budget, "seed": seed,
                                    "adversary_data": adv, "target_family": tgt,
                                })
    return cells


def _cell_key(c: dict) -> tuple:
    return (c["model"], c["recipe"], c["lora_rank"], c["budget_n"], c["seed"],
            c["adversary_data"], c["target_family"])


def _load_done(out_csv: Path) -> set:
    done = set()
    if out_csv.exists():
        with open(out_csv) as fh:
            for row in csv.DictReader(fh):
                done.add((
                    row["model"], row["recipe"],
                    # tolerate float-formatted ranks ("8.0") from pandas-written CSVs, not just "8"
                    None if row["lora_rank"] in ("", "None", "nan", "NaN") else int(float(row["lora_rank"])),
                    int(row["budget_n"]), int(row["seed"]),
                    row["adversary_data"], row["target_family"],
                ))
    return done


def run_sweep(
    cfg,
    paths: DataPaths,
    model_paths: Dict[str, str],
    out_csv: str,
    attack_cfgs: Optional[Dict[str, object]] = None,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = True,
    tracker=None,
) -> str:
    """Execute the sweep (or a shard of it). ``model_paths`` maps 'A'/'B'/'A_prime'
    to a checkpoint dir or HF id. ``attack_cfgs`` maps each recipe ('lora','full_ft')
    to its config node so the recipe-specific hyperparameters (e.g. LoRA lr 1e-4 vs
    full-FT lr 1e-5) are used; if None, only ``cfg.attack``'s own recipe is available.
    Returns the CSV path written."""
    from src.models.load import load_tokenizer
    from src.train.attack import run_attack
    from src.utils.io import save_json

    cells = build_cells(cfg.sweep)
    if num_shards > 1:
        cells = cells[shard_index::num_shards]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    history_path = out_csv.with_name(out_csv.stem + "_history.json")
    history_all: Dict[str, list] = {}
    # Global resume: skip any cell already completed in ANY shard file, so re-runs
    # (even with a different shard count) never recompute finished work.
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
    homolog_ratio = float(cfg_get(cfg, "homolog_attack_ratio", 1.0))
    background = []
    if Path(paths.background).exists():
        from src.utils.io import fasta_seqs

        background = fasta_seqs(paths.background)

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
            seqs = attacker_trainset(paths, c["target_family"], c["adversary_data"],
                                     c["budget_n"], seed=c["seed"], homolog_ratio=homolog_ratio)
            test_seqs = target_test_seqs(paths, c["target_family"])
            res = run_attack(
                checkpoint_path=model_paths[c["model"]],
                tokenizer=tokenizer,
                attacker_seqs=seqs,
                target_test_seqs=test_seqs,
                attack_cfg=attack_cfg,
                device=device,
                seed=c["seed"],
                max_length=max_length,
                background_seqs=background,
                ppl_batch_size=eval_bs,
                eval_masks_per_pass=eval_mpp,
                max_eval_seqs=n_eval,
                eval_every=eval_ev,
                tracker=tracker,
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
            history_all["|".join(str(x) for x in _cell_key(c))] = res["history"]
            save_json(history_all, history_path)
            if tracker:
                tracker.log({k: row[k] for k in ("best_target_ppl", "budget_n", "model")})
            print(f"[sweep] {_cell_key(c)} -> best_ppl={res['best_target_ppl']:.4f}")
    return str(out_csv)


def _attack_cfg_for(attack_cfg, recipe: str, lora_rank):
    """Clone a recipe's attack config node and override mode/rank for this cell."""
    from omegaconf import OmegaConf

    node = OmegaConf.create(OmegaConf.to_container(attack_cfg, resolve=True))
    node.mode = recipe
    if recipe == "lora" and lora_rank is not None:
        node.lora_rank = int(lora_rank)
    return node
