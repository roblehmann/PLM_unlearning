"""Phase 5 — compute-matched control -> checkpoint A'.

From the SAME base checkpoint, run plain MLM continued-pretraining on R for the same
estimated FLOP budget B consumed (matched on FLOPs, not wall-clock). A' is the
headline denominator: it isolates the safeguard from "extra training".

    python scripts/04_train_control.py model=esm2_35m

Done when: A' exists; its consumed compute matches B's within a logged tolerance.
"""
from pathlib import Path

import _bootstrap  # noqa: F401
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 5 — TRAIN COMPUTE-MATCHED CONTROL A'")
    from src.data.datamodule import DataPaths
    from src.models.load import load_mlm, load_tokenizer, save_model
    from src.train.mlm import continued_pretrain
    from src.utils.flops import relative_gap
    from src.utils.io import ensure_dir, fasta_seqs, load_json, save_json
    from src.utils.seeding import set_seed

    set_seed(cfg.seed)
    paths = DataPaths(cfg.paths.data)
    device = cfg.device
    max_len = int(cfg.data.max_length)
    ckpt_Ap = str(Path(cfg.paths.checkpoints) / "A_prime")
    tracker = _bootstrap.init_tracker_from_cfg(cfg, run_name="train_control")

    b_record = load_json(f"{cfg.paths.results}/B_compute.json")
    target_flops = float(b_record["flops_total"])
    print(f"matching A' to B's FLOPs: target = {target_flops:.3e}")

    tok = load_tokenizer(cfg.model.name)
    base = load_mlm(cfg.model.name, device=device)
    retain = fasta_seqs(paths.retain_train)

    out = continued_pretrain(
        base, tok, retain, device,
        lr=float(cfg.blocking.eta_outer),            # same outer LR scale as B's defender
        batch_size=int(cfg.blocking.retain_batch),
        mlm_probability=float(cfg.blocking.mlm_probability),
        max_length=max_len,
        target_flops=target_flops,
        seed=cfg.seed,
        warmup_steps=int(cfg.blocking.warmup_steps),
        tracker=tracker,
    )
    ensure_dir(cfg.paths.checkpoints)
    save_model(out["model"], ckpt_Ap)

    realized = out["flops"]["total_flops"]
    gap = relative_gap(realized, target_flops)
    record = {
        "checkpoint": ckpt_Ap,
        "target_flops": target_flops,
        "realized_flops": realized,
        "flop_match_rel_gap": gap,
        "steps": out["steps"],
        "wall_clock_s": out["flops"]["wall_clock_s"],
    }
    save_json(record, f"{cfg.paths.results}/Aprime_compute.json")
    tracker.log_summary(record)
    tracker.finish()

    _bootstrap.banner("CONTROL A' DONE")
    print(f"  realized FLOPs={realized:.3e}  target={target_flops:.3e}  rel_gap={gap*100:.2f}% "
          f"-> {'OK' if gap < 0.05 else 'WIDEN tolerance or adjust steps'}")


if __name__ == "__main__":
    main()
