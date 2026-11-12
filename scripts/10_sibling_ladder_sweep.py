"""Phase 10 — sibling-ladder relearning sweep over {B, A'} × recipes × budgets × seeds
× ladder rungs.

Fine-tune each checkpoint on a same-clan SIBLING family of F at increasing homology
distance (data/ladder/<F>/corpora/<rung_acc>_ft.fasta; never F itself) and score the
held-out FORGET-family pseudo-perplexity (F-PPL on data/splits/F_test.fasta). Tests
whether tuning on a near neighbour of F reawakens the blocked capability by proximity,
and at which rung that stops.

    # single process:
    python scripts/10_sibling_ladder_sweep.py sweep=sibling_ladder model=esm2_35m
    # SLURM array (set --array=0-15): each task runs one shard, writes its own CSV.

The rung list + distance metadata come from data/ladder/<F>/ladder_selected.json. Only
B and A_prime are swept (A, the base model, was never blocked so has nothing to
reawaken). Analyse with scripts/10b_sibling_ladder_analyze.py.

Done when: the grid completes, persisted as a tidy long-format table (one row per model
× recipe × rank × budget × seed × rung).
"""
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 10 — SIBLING-LADDER RELEARNING SWEEP")
    from omegaconf import OmegaConf

    from src.sweep.run_ladder_sweep import build_ladder_cells, load_rungs, run_ladder_sweep
    from src.utils.io import ensure_dir

    ladder_dir = os.path.join(cfg.paths.data, "ladder", cfg.data.forget_family.pfam_id)
    if not os.path.isdir(ladder_dir):
        raise SystemExit(f"No ladder dir at {ladder_dir}. Expected data/ladder/<F>/ from Phase-1 prep.")

    attack_cfgs = {
        "lora": OmegaConf.load(os.path.join(_bootstrap.CONFIGS, "attack", "lora.yaml")),
        "full_ft": OmegaConf.load(os.path.join(_bootstrap.CONFIGS, "attack", "full_ft.yaml")),
    }
    model_paths = {
        "B": os.path.join(cfg.paths.checkpoints, "B"),
        "A_prime": os.path.join(cfg.paths.checkpoints, "A_prime"),
    }

    # shard from the SLURM array if present, else a single shard
    shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", cfg.get("shard_index", 0) or 0))
    num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", cfg.get("num_shards", 1) or 1))
    only = cfg.sweep.get("rungs", None)
    rungs = load_rungs(ladder_dir, list(only) if only else None)
    n_cells = len(build_ladder_cells(cfg.sweep, rungs))
    print(f"ladder {ladder_dir}: {len(rungs)} rungs; grid has {n_cells} cells; "
          f"this is shard {shard_index}/{num_shards}")

    ensure_dir(os.path.join(cfg.paths.results, "sibling_ladder"))
    out_csv = os.path.join(cfg.paths.results, "sibling_ladder",
                           f"sweep_shard{shard_index}_of{num_shards}.csv")
    tracker = _bootstrap.init_tracker_from_cfg(cfg, run_name=f"ladder_shard{shard_index}")
    run_ladder_sweep(cfg, ladder_dir, model_paths, out_csv, attack_cfgs=attack_cfgs,
                     shard_index=shard_index, num_shards=num_shards, tracker=tracker)
    tracker.finish()
    _bootstrap.banner(f"LADDER SWEEP SHARD {shard_index} DONE -> {out_csv}")


if __name__ == "__main__":
    main()
