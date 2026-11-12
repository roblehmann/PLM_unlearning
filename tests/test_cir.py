"""Tests for src.train.cir (CIR safeguard).

Pure-logic pieces (special-token set, layer resolution) run with no deps. The
collapse/PCA/projection math needs torch (importorskip). The end-to-end micro-run is
gated on transformers + a modern torch like the taskblock micro-run."""
from types import SimpleNamespace

import pytest

from src.train.cir import resolve_layers, special_id_set


# --------------------------------------------------------------------------- #
# Pure logic (no torch)
# --------------------------------------------------------------------------- #
def test_special_id_set_collects_and_drops_none():
    tok = SimpleNamespace(cls_token_id=0, eos_token_id=2, pad_token_id=1, mask_token_id=32)
    assert special_id_set(tok) == {0, 1, 2, 32}
    tok2 = SimpleNamespace(cls_token_id=0, eos_token_id=2, pad_token_id=1, mask_token_id=None)
    assert special_id_set(tok2) == {0, 1, 2}


def test_resolve_layers_clamps_and_falls_back():
    m12 = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=12))
    assert resolve_layers(m12, [3, 4, 5, 6, 7, 8]) == [3, 4, 5, 6, 7, 8]
    # out-of-range indices for a 6-layer model are dropped
    m6 = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=6))
    got = resolve_layers(m6, [3, 4, 5, 6, 7, 8])
    assert got == [3, 4, 5]
    # empty band -> non-empty middle fallback, all in range
    fb = resolve_layers(m6, [])
    assert fb and all(0 <= i < 6 for i in fb)


# --------------------------------------------------------------------------- #
# Collapse / PCA / projection math (torch)
# --------------------------------------------------------------------------- #
def test_collapse_is_orthogonal_to_basis():
    torch = pytest.importorskip("torch")
    from src.train.cir import collapse

    torch.manual_seed(0)
    dim, T = 8, 64
    X = torch.randn(T, dim) + 3.0           # nonzero mean
    mean = X.mean(0)
    M = torch.randn(dim, 2)
    Q, _ = torch.linalg.qr(M)               # orthonormal columns
    basis = Q[:, :2].t().contiguous()       # [2, dim] orthonormal rows
    pure = collapse(X, mean, basis, collapse_mean=True)
    assert pure.shape == X.shape
    # purified vectors carry no component along the collapsed directions
    assert pure.matmul(basis.t()).abs().max().item() < 1e-4


def test_top_k_basis_recovers_dominant_axis():
    torch = pytest.importorskip("torch")
    from src.train.cir import _Cov, _top_k_basis

    torch.manual_seed(0)
    dim = 6
    X = torch.randn(300, dim) * 0.1
    X[:, 0] = X[:, 0] + torch.randn(300) * 5.0   # variance dominated by axis 0
    cov_obj = _Cov(dim, "cpu", torch.float32)
    cov_obj.add(X)
    cov, mean = cov_obj.cov_mean()
    basis, evals = _top_k_basis(cov, mean, 1)
    assert basis.shape == (1, dim)
    assert abs(basis[0, 0].item()) > 0.9                       # aligned with axis 0
    assert abs((basis[0] @ basis[0]).item() - 1.0) < 1e-4      # orthonormal row


def test_top_k_basis_zero_k_is_empty():
    torch = pytest.importorskip("torch")
    from src.train.cir import _Cov, _top_k_basis

    cov_obj = _Cov(5, "cpu", torch.float32)
    cov_obj.add(torch.randn(20, 5))
    cov, mean = cov_obj.cov_mean()
    basis, _ = _top_k_basis(cov, mean, 0)
    assert basis.shape == (0, 5)


def test_discriminative_filter_keeps_F_distinctive():
    torch = pytest.importorskip("torch")
    from src.train.cir import _discriminative_filter

    dim = 4
    basis = torch.eye(dim)[:2]                                  # candidate rows e0, e1
    cov_R = torch.diag(torch.tensor([5.0, 0.1, 0.0, 0.0]))      # e0 shared (high in R)
    cov_F = torch.diag(torch.tensor([0.2, 9.0, 0.0, 0.0]))      # e1 distinctive (high in F)
    collapse_rows = _discriminative_filter(basis, cov_R, cov_F, keep_ratio=2.0)
    # e0 collapsed (var_F 0.2 <= 2*5); e1 kept (var_F 9 > 2*0.1) -> only e0 returned
    assert collapse_rows.shape[0] == 1
    assert abs(collapse_rows[0, 0].item() - 1.0) < 1e-6


def test_projected_l2_splits_kept_vs_collapsed():
    torch = pytest.importorskip("torch")
    from src.analysis.weightspace import projected_l2_by_group

    out_f, in_f = 4, 5
    key = "esm.encoder.layer.5.output.dense.weight"
    W0 = torch.zeros(out_f, in_f)
    a0 = torch.zeros(in_f); a0[0] = 1.0          # collapsed input direction
    g0 = torch.zeros(out_f); g0[0] = 1.0         # collapsed output direction
    bases = {5: {"act_basis": a0.unsqueeze(0).numpy(), "grad_basis": g0.unsqueeze(0).numpy()}}

    # ΔW entirely inside the collapsed subspace -> kept ≈ 0
    dW_coll = torch.outer(g0, a0) * 3.0
    res = projected_l2_by_group({key: W0}, {key: W0 + dW_coll}, bases)
    assert res["__total__"]["kept_l2"] < 1e-5
    assert abs(res["__total__"]["total_l2"] - 3.0) < 1e-4

    # ΔW entirely in the kept subspace -> collapsed ≈ 0
    a1 = torch.zeros(in_f); a1[3] = 1.0
    g1 = torch.zeros(out_f); g1[2] = 1.0
    dW_keep = torch.outer(g1, a1) * 2.0
    res2 = projected_l2_by_group({key: W0}, {key: W0 + dW_keep}, bases)
    assert abs(res2["__total__"]["kept_l2"] - 2.0) < 1e-4
    assert res2["__total__"]["collapsed_l2"] < 1e-5


# --------------------------------------------------------------------------- #
# End-to-end micro-run (needs transformers + modern torch; skips otherwise)
# --------------------------------------------------------------------------- #
def _maybe_skip_env():
    pytest.importorskip("transformers")
    import torch

    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 0):
        pytest.skip("needs a modern torch for ESM-2")


@pytest.mark.slow
def test_cir_micro_run(built_data):
    """One epoch / two steps of CIR on the 8M model over toy sequences. Asserts the
    train_taskblock-compatible contract dict + the collapse-subspace bases come back."""
    _maybe_skip_env()
    from omegaconf import OmegaConf

    from src.models.load import load_mlm, load_tokenizer
    from src.train.cir import train_cir
    from src.utils.io import fasta_seqs

    cfg = OmegaConf.create({
        "subspace_mode": "retain_pca", "n_act_pcs": 2, "n_grad_pcs": 2, "collapse_mean": True,
        "unlearn_rate": 1e-2, "layer_band": [2, 3], "eta_outer": 5e-5, "retain_batch": 4,
        "forget_batch": 4, "warmup_steps": 1, "grad_clip": 1.0, "mlm_probability": 0.15,
        "pc_corpus_seqs": 8, "pc_batch": 4, "pc_act_position_policy": "all_unmasked",
        "pc_grad_mask_draws": 1, "recompute_pcs_every_epoch": True, "discriminative_keep_ratio": 2.0,
        "max_epochs": 1, "steps_per_epoch": 2, "terminate_retain_ppl_ratio": 100.0, "log_every": 1,
    })
    tok = load_tokenizer("facebook/esm2_t6_8M_UR50D")
    base = load_mlm("facebook/esm2_t6_8M_UR50D", device="cpu")
    out = train_cir(
        base, tok,
        retain_seqs=fasta_seqs(built_data.paths.retain_train)[:8],
        forget_seqs=fasta_seqs(built_data.paths.F_train)[:8],
        cfg=cfg, device="cpu", seed=0, max_length=64,
        retain_eval_seqs=fasta_seqs(built_data.paths.retain_holdout)[:4],
        eval_batch_size=4, eval_masks_per_pass=4, eval_max_seqs=4,
    )
    assert "model" in out
    assert out["flops"]["total_flops"] > 0
    assert out["flops"]["overhead_factor"] > 0
    assert out["layers"] == [2, 3]
    assert set(out["subspace"].keys()) == {2, 3}
    b = out["subspace"][2]
    assert b["act_basis"].shape[0] <= 2 and b["grad_basis"].shape[0] <= 2
