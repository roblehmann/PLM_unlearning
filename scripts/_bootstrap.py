"""Shared bootstrap for the phase scripts: put the project root on sys.path (so
`import src.*` works without `pip install -e .`), default TASKBLOCK_ROOT, and offer
tiny helpers for tracker + data paths."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("TASKBLOCK_ROOT", ROOT)
CONFIGS = os.path.join(ROOT, "configs")


def init_tracker_from_cfg(cfg, run_name: str):
    from omegaconf import OmegaConf

    from src.utils.tracking import init_tracker

    return init_tracker(
        backend=cfg.tracking.backend,
        project=cfg.tracking.project,
        run_name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        results_dir=cfg.paths.results,
        mode=cfg.tracking.get("mode", "online"),
    )


def data_paths(cfg):
    from src.data.datamodule import DataPaths

    return DataPaths(cfg.paths.data)


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
