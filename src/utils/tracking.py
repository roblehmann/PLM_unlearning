"""Experiment tracking shim.

`init_tracker` returns a wandb-backed tracker when configured and importable, and
otherwise an offline JSONL logger writing under ``<results>/runs/<name>.jsonl``.
This keeps smoke tests and HPC batch jobs runnable with no network and no wandb.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class Tracker:
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def log_summary(self, data: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def finish(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()


class JSONLTracker(Tracker):
    """Offline logger. One JSON object per line; a sidecar ``.summary.json``."""

    def __init__(self, run_name: str, results_dir: str, config: Optional[Dict[str, Any]] = None):
        self.dir = Path(results_dir) / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_name}.jsonl"
        self.summary_path = self.dir / f"{run_name}.summary.json"
        self.summary: Dict[str, Any] = {}
        self._fh = open(self.path, "a")
        if config is not None:
            self._fh.write(json.dumps({"_event": "config", "config": config, "ts": time.time()}, default=str) + "\n")
            self._fh.flush()

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        rec = dict(data)
        if step is not None:
            rec["_step"] = step
        rec["ts"] = time.time()
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def log_summary(self, data: Dict[str, Any]) -> None:
        self.summary.update(data)
        with open(self.summary_path, "w") as fh:
            json.dump(self.summary, fh, indent=2, default=str)

    def finish(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class WandbTracker(Tracker):
    def __init__(self, project: str, run_name: str, config: Optional[Dict[str, Any]], mode: str):
        import wandb  # imported lazily; absence triggers the JSONL fallback in init_tracker

        self._wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config, mode=mode, reinit=True)

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        self.run.log(data, step=step)

    def log_summary(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            self.run.summary[k] = v

    def finish(self) -> None:
        self.run.finish()


def init_tracker(
    backend: str,
    project: str,
    run_name: str,
    config: Optional[Dict[str, Any]],
    results_dir: str,
    mode: str = "online",
) -> Tracker:
    """Build a tracker. Falls back to JSONL on backend='none'/'mlflow-unavailable'
    or if wandb cannot be imported/initialised."""
    backend = (backend or "none").lower()
    if backend == "wandb" and mode != "disabled":
        try:
            return WandbTracker(project=project, run_name=run_name, config=config, mode=mode)
        except Exception as e:  # missing wandb, no login, offline node, etc.
            print(f"[tracking] wandb unavailable ({e!r}); using offline JSONL logger.")
    return JSONLTracker(run_name=run_name, results_dir=results_dir, config=config)
