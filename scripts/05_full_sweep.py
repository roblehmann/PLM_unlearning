"""Phase 6 — full attacker sweep over {A, B, A'} × recipes × budgets × seeds, plus
the two durability axes (homolog_augmented adversary data; target_family F').

    # single process:
    python scripts/05_full_sweep.py model=esm2_35m
    # SLURM array (set --array=0-15): each task runs one shard, writes its own CSV.

    # retain-only RELEARNING probe (opt-in): fine-tune on R alone, zero F, watch F-PPL
    python scripts/05_full_sweep.py model=esm2_35m sweep=relearn

The relearning probe rides the adversary_data axis as the value 'retain_only': the
attacker trains purely on the retain set R (zero forget examples) while F-PPL is still
scored on the held-out forget family. A collapse of B's F-PPL toward A's floor is the
dormant-not-deleted signal. Its rows append into results/sweep/ and analyze (06) picks
them up as just another adversary_data condition; see configs/sweep/relearn.yaml.

Done when: the full grid completes, persisted as a tidy long-format table (one row
per model × recipe × rank × budget × seed × adversary_data × target_family).
"""
import os

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 6 — FULL SWEEP")
    from omegaconf import OmegaConf

    from src.data.datamodule import DataPaths
    from src.sweep.run_sweep import build_cells, run_sweep
    from src.utils.io import ensure_dir

    paths = DataPaths(cfg.paths.data)
    attack_cfgs = {
        "lora": OmegaConf.load(os.path.join(_bootstrap.CONFIGS, "attack", "lora.yaml")),
        "full_ft": OmegaConf.load(os.path.join(_bootstrap.CONFIGS, "attack", "full_ft.yaml")),
    }
    model_paths = {
        "A": cfg.model.name,  # base ESM-2 (HF id loads fine as a checkpoint)
        "B": os.path.join(cfg.paths.checkpoints, "B"),
        "A_prime": os.path.join(cfg.paths.checkpoints, "A_prime"),
    }

    # shard from the SLURM array if present, else a single shard
    shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", cfg.get("shard_index", 0) or 0))
    num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", cfg.get("num_shards", 1) or 1))
    n_cells = len(build_cells(cfg.sweep))
    print(f"grid has {n_cells} cells; this is shard {shard_index}/{num_shards}")

    ensure_dir(os.path.join(cfg.paths.results, "sweep"))
    out_csv = os.path.join(cfg.paths.results, "sweep", f"sweep_shard{shard_index}_of{num_shards}.csv")
    tracker = _bootstrap.init_tracker_from_cfg(cfg, run_name=f"sweep_shard{shard_index}")
    run_sweep(cfg, paths, model_paths, out_csv, attack_cfgs=attack_cfgs,
              shard_index=shard_index, num_shards=num_shards, tracker=tracker)
    tracker.finish()
    _bootstrap.banner(f"SWEEP SHARD {shard_index} DONE -> {out_csv}")


if __name__ == "__main__":
    main()
