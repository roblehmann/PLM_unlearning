"""Phase 0 — scaffold & smoke.

Loads the (8M) model, tokenizes a handful of sequences, runs one forward pass and
one MLM gradient step, exercises pseudo-perplexity, and logs to the tracker.

    python scripts/00_smoke.py model=esm2_8m tracking.backend=none

Done when: a single MLM step runs end-to-end and appears in the tracker; configs
resolve via Hydra.
"""
import _bootstrap  # noqa: F401  (sys.path + env)
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 0 — SMOKE")
    print(OmegaConf.to_yaml(cfg))  # proves configs resolve

    import torch

    from src.data.datamodule import make_mlm_loader
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.models.load import build_model
    from src.train.mlm import mlm_loss
    from src.utils.io import get_device
    from src.utils.seeding import set_seed

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"device = {device}; model = {cfg.model.name}")

    model, tok = build_model(cfg.model, device=device)
    seqs = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF",
    ]
    loader = make_mlm_loader(seqs, tok, batch_size=2, mlm_probability=0.15, max_length=64, shuffle=False, seed=cfg.seed)
    batch = next(iter(loader))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    opt.zero_grad(set_to_none=True)
    loss = mlm_loss(model, batch, device)
    loss.backward()
    opt.step()
    print(f"one MLM step: loss = {loss.item():.4f}")

    ppl = pseudo_perplexity(model, tok, seqs, device, batch_size=8, max_length=64, masks_per_pass=1)
    print(f"pseudo-perplexity (toy seqs) = {ppl.ppl:.3f}")

    tracker = _bootstrap.init_tracker_from_cfg(cfg, run_name="smoke")
    tracker.log({"smoke_mlm_loss": float(loss.item()), "smoke_ppl": ppl.ppl}, step=1)
    tracker.log_summary({"smoke_ok": True})
    tracker.finish()
    _bootstrap.banner("SMOKE OK — one MLM step ran and logged")


if __name__ == "__main__":
    main()
