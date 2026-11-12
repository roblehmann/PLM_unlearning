"""Shared plumbing for experiment 4 — "distinctive footprint vs F_test generator".

THE CLAIM UNDER TEST
--------------------
CIR blocks the in-distribution Pfam family (6.4x) but not the OOD metagenomic family
(<=2.8x), even though the two have near-identical O(F,R) removable subspaces (0.672 vs
0.649). The proposed mechanism is a *dissociation*:

    footprint  := the parameter directions that continued pretraining on F_train writes,
                  i.e. what is distinctive about the acquired model;
    generator  := the directions that regenerate *homology-decontaminated* F_test.

For the Pfam family ESM-2 already carries a generalising representation from UniRef50,
so acquisition sharpens it: footprint == generator, and CIR's collapse of the footprint
also collapses the generator -> a block. For the metagenomic family ESM-2 has no
dedicated representation, so acquisition must write a large, distinctive, *fresh*
structure — but because F_test is homology-decontaminated that structure is largely
memorisation of F_train. Whatever F_test competence exists rides the general covariation
machinery R. CIR finds and collapses the footprint (it moves a lot of weight: 7.4% rel
||dtheta||, 16% in one layer) and F_test PPPL does not move. No block.

The reportable inversion: the *recent, OOD, freshly-written* family is the HARDER one to
remove — the opposite of "a thin recent perturbation is easy to delete". Recent and
distinctive is not the same as load-bearing.

Corollary the tests also check: O(F,R) as currently measured captures the footprint (which
includes memorisation), not the generator — so it is the right predictor for Pfam and the
wrong one for MGnify, which is exactly why it mispredicted.

WHAT THIS MODULE PROVIDES
-------------------------
* :data:`ARMS` — the two families as parallel arms over the *existing* run roots, so
  nothing here re-derives data, tau, or checkpoints that the main pipeline already owns.
* One common space for every subspace comparison. Every basis in this experiment lives in
  the **down-projection input space** of ``esm.encoder.layer[i].output.dense`` (the
  post-GELU FFN activations, dim 1920 at 35M) for the CIR layer band [3..8]. That is the
  space CIR's collapse acts in, the space Delta_acquire's row space lives in, and the space
  the F_test loss gradient lives in — so footprint, CIR target and generator are directly
  commensurable and every overlap is an apples-to-apples number.
* :func:`acquire` — theta_acq = theta_0 continued-pretrained on F_train. NOTE this is a
  fixed-budget run with **no F_test-based early stopping**: selecting the checkpoint on
  F_test would leak the very held-out set whose train/test gap is the measurement.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (plm_unlearning/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TASKBLOCK_ROOT", str(ROOT))

# A smoke/scratch run sets TB_FVG_TAG so its artifacts and its throwaway theta_acq land
# beside — never on top of — the real ones.
TAG = os.environ.get("TB_FVG_TAG", "").strip()
_SUFFIX = f"_{TAG}" if TAG else ""
RESULTS = HERE / f"results{_SUFFIX}"

# The headline 35M model every arm was run at.
MODEL_ID = "facebook/esm2_t12_35M_UR50D"

# CIR's targeted band (configs/blocking/cir.yaml). Everything in this experiment is
# measured inside it, because that is where CIR is allowed to edit.
LAYER_BAND = [3, 4, 5, 6, 7, 8]

# Acquisition protocol. Identical for both arms — the arms must differ only in the
# family. 8192/40 mirrors the pipeline's PPL_floor protocol (configs/config.yaml
# `floor:`), so theta_acq is the same object the floor number was measured on, except
# that we now keep the weights.
N_ACQUIRE = 8192
ACQUIRE_EPOCHS = 40
ACQUIRE_LR = 1.0e-5
ACQUIRE_BATCH = 16
ACQUIRE_SEED = 0

# Scoring. Matches configs/config.yaml `eval:` so PPPLs here are comparable to every
# other number in the project (tau, the sweep, thresholds.json).
N_EVAL = 256
# Sequences per corpus for the in-training probe curve. Smaller than N_EVAL on purpose:
# the curve is diagnostic (it shows the gap opening up over epochs) and never selects
# weights, so it does not need the headline set's precision — but it is scored ~10 times
# during acquisition, so its cost is what would otherwise dominate the run.
PROBE_N = 64
EVAL_BATCH = 64
MASKS_PER_PASS = 8
MAX_LENGTH = 1024

# Subspace dimension used for every basis, matching CIR's n_act_pcs=24.
N_PCS = 24


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    key: str
    label: str
    family: str
    kind: str                     # 'in_distribution' | 'ood'
    run_root: Path                # existing pipeline run root (data/, checkpoints/, results/)
    cir_b: Path                   # the CIR-blocked B whose edit we compare against
    cir_label: str
    cir_forget_ratio: float       # static block achieved there, B/base F-PPL
    note: str = ""

    # ---- derived paths ---- #
    @property
    def data(self) -> Path:
        return self.run_root / "data"

    @property
    def acquired_ckpt(self) -> Path:
        """theta_acq lives with the run it belongs to, not in the experiment dir."""
        return self.run_root / "checkpoints" / f"acquired_F{_SUFFIX}"

    @property
    def results(self) -> Path:
        d = RESULTS / self.key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def thresholds(self) -> dict:
        p = self.run_root / "results" / "thresholds.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def cir_overlap(self) -> dict:
        p = self.run_root / "results" / "cir_overlap.json"
        return json.loads(p.read_text()) if p.exists() else {}


ARMS: Dict[str, Arm] = {
    # The in-distribution control: a real Pfam family (SDR, clan CL0063) that base ESM-2
    # already represents from UniRef50. This is the run where the SAME CIR code produced a
    # 6.41x static block (results/cir35m_peak/B_compute.json).
    "pfam": Arm(
        key="pfam",
        label="Pfam PF00106 (SDR) — in-distribution",
        family="PF00106",
        kind="in_distribution",
        run_root=ROOT,
        cir_b=ROOT / "checkpoints" / "cir35m_peak" / "B",
        cir_label="cir35m_peak (r0.2_t1.01)",
        cir_forget_ratio=6.41,
        note="F_test here is the ORIGINAL split, which is known to share homology with "
             "F_train (see the headroom-collapse diagnosis). t0_decontaminate.py applies "
             "the SAME mmseqs criterion to both arms so the train/test gap is not "
             "confounded by that leakage.",
    ),
    # The OOD arm: a metagenomic family with no Pfam HMM and no dedicated ESM-2
    # representation (29.5% headroom, the only substrate in the project with real
    # headroom). Here CIR tops out at 2.78x across two orthogonal knob sweeps.
    "mgnify": Arm(
        key="mgnify",
        label="MGnify MGYF0000018690 — OOD metagenomic",
        family="MGYF0000018690",
        kind="ood",
        run_root=ROOT / "runs" / "MGYF18690_metagenomic",
        cir_b=ROOT / "runs" / "MGYF18690_metagenomic" / "checkpoints" / "cir_knob"
              / "retain_pca_n24_b3-8" / "B",
        cir_label="cir_knob best (retain_pca_n24_b3-8, r0.4_t1.02)",
        cir_forget_ratio=2.78,
        note="Source-aware split; 0 F_test/F_train hits at -c0.8, residual partial "
             "homology at -c0.5 (~30% of F_test).",
    ),
}

ARM_ORDER = ["pfam", "mgnify"]


def get_arm(key: str) -> Arm:
    if key not in ARMS:
        raise SystemExit(f"unknown arm {key!r}; known: {', '.join(ARM_ORDER)}")
    return ARMS[key]


# --------------------------------------------------------------------------- #
# Evaluation sets
# --------------------------------------------------------------------------- #
#   F_train_seen   : the exact sequences theta_acq was trained on (memorisation probe)
#   F_train_unseen : F_train members NOT in the acquisition subset — same distribution,
#                    same homology neighbourhood, never gradient-updated on. Isolates
#                    "memorised these sequences" from "memorised this train homology cloud".
#   F_test         : the pipeline's held-out split (arm-native decontamination)
#   F_test_decon   : F_test after the identical cross-arm mmseqs filter (t0)
#   retain_holdout : benign utility reference
EVAL_SET_NAMES = ["F_train_seen", "F_train_unseen", "F_test", "F_test_decon", "retain_holdout"]


def acquisition_seqs(arm: Arm, n: int = N_ACQUIRE, seed: int = ACQUIRE_SEED) -> List[str]:
    """The exact F_train subset theta_acq is trained on. Deterministic in ``seed``."""
    from src.data.datamodule import subsample
    from src.utils.io import fasta_seqs

    return subsample(fasta_seqs(arm.data / "splits" / "F_train.fasta"), n, seed=seed)


def eval_sets(arm: Arm, n_eval: int = N_EVAL, seed: int = 0) -> Dict[str, List[str]]:
    """The five scoring corpora, each capped at ``n_eval`` sequences."""
    from src.data.datamodule import subsample
    from src.utils.io import fasta_seqs

    splits = arm.data / "splits"
    seen = acquisition_seqs(arm)
    seen_set = set(seen)
    f_train = fasta_seqs(splits / "F_train.fasta")
    unseen = [s for s in f_train if s not in seen_set]

    decon_path = arm.results / "F_test_decon.fasta"
    sets = {
        "F_train_seen": subsample(seen, n_eval, seed=seed),
        "F_train_unseen": subsample(unseen, n_eval, seed=seed),
        "F_test": subsample(fasta_seqs(splits / "F_test.fasta"), n_eval, seed=seed),
        "retain_holdout": subsample(fasta_seqs(splits / "retain_holdout.fasta"), n_eval, seed=seed),
    }
    if decon_path.exists():
        sets["F_test_decon"] = subsample(fasta_seqs(decon_path), n_eval, seed=seed)
    return sets


def primary_test_key(sets: Dict[str, List[str]], min_seqs: int = 32) -> str:
    """The decontaminated F_test if t0 produced a usable one, else the native split.

    Every headline number in this experiment is reported against this key, because the
    whole mechanism claim is about *decontaminated* F_test — a gap measured against a
    leaky held-out set would be the confound, not the finding.
    """
    d = sets.get("F_test_decon")
    return "F_test_decon" if d is not None and len(d) >= min_seqs else "F_test"


# --------------------------------------------------------------------------- #
# Models / scoring
# --------------------------------------------------------------------------- #
def load_base(device: str):
    from src.models.load import load_checkpoint

    return load_checkpoint(MODEL_ID, device=device).eval()


def load_acquired(arm: Arm, device: str):
    from src.models.load import load_checkpoint

    if not (arm.acquired_ckpt / "config.json").exists():
        raise SystemExit(
            f"missing theta_acq for arm {arm.key} at {arm.acquired_ckpt} — run t1_acquire_gap.py first")
    return load_checkpoint(str(arm.acquired_ckpt), device=device).eval()


def tokenizer():
    from src.models.load import load_tokenizer

    return load_tokenizer(MODEL_ID)


def ppl(model, tok, seqs: Sequence[str], device: str) -> float:
    from src.eval.pseudo_perplexity import pseudo_perplexity

    if not seqs:
        return float("nan")
    return pseudo_perplexity(model, tok, seqs, device, EVAL_BATCH, MAX_LENGTH,
                             masks_per_pass=MASKS_PER_PASS).ppl


def ppl_all(model, tok, sets: Dict[str, Sequence[str]], device: str, prefix: str = "") -> Dict[str, float]:
    out = {}
    for name, seqs in sets.items():
        out[name] = ppl(model, tok, seqs, device)
        print(f"  {prefix}{name:<16} n={len(seqs):>4}  PPPL={out[name]:.4f}", flush=True)
    return out


def layers_for(model) -> List[int]:
    from src.train.cir import resolve_layers

    return resolve_layers(model, LAYER_BAND)


def dense(model, li: int):
    from src.train.cir import _dense

    return _dense(model, li)


# --------------------------------------------------------------------------- #
# Acquisition (theta_acq)
# --------------------------------------------------------------------------- #
def acquire(arm: Arm, device: str, *, epochs: int = ACQUIRE_EPOCHS, n_train: int = N_ACQUIRE,
            lr: float = ACQUIRE_LR, batch_size: int = ACQUIRE_BATCH, seed: int = ACQUIRE_SEED,
            eval_every: int = 4, probe_n: int = PROBE_N, max_length: int = MAX_LENGTH) -> dict:
    """theta_0 -> theta_acq by plain MLM continued pretraining on F_train.

    Fixed budget, **no early stopping on F_test**. The whole point of test 1 is the
    train/test gap; choosing the checkpoint by F_test would select against exactly the
    gap we are trying to measure. The per-epoch probe curve is recorded (a 128-seq
    subsample of each of F_train_seen / F_test) but never used to select weights.
    """
    import torch

    from src.data.datamodule import make_mlm_loader, subsample
    from src.models.load import load_checkpoint, save_model
    from src.train.mlm import batch_tokens, mlm_loss
    from src.utils.flops import FlopCounter
    from src.utils.seeding import set_seed

    set_seed(seed)
    tok = tokenizer()
    train_seqs = acquisition_seqs(arm, n_train, seed)
    sets = eval_sets(arm, probe_n, seed=seed)
    probe = {k: sets[k] for k in ("F_train_seen", primary_test_key(sets)) if k in sets}

    model = load_checkpoint(MODEL_ID, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    counter = FlopCounter(n_params=sum(p.numel() for p in params))
    loader = make_mlm_loader(train_seqs, tok, batch_size, 0.15, max_length, shuffle=True, seed=seed)

    def probe_now(epoch: int) -> dict:
        rec = {"epoch": epoch}
        for name, seqs in probe.items():
            rec[name] = ppl(model.eval(), tok, seqs, device)
        model.train()
        rec["gap_ratio"] = (rec.get(primary_test_key(sets), float("nan"))
                            / rec.get("F_train_seen", float("nan")))
        print(f"  [acquire] epoch {epoch:>3}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"), flush=True)
        return rec

    history = [probe_now(0)]
    model.train()
    for epoch in range(1, epochs + 1):
        tot, nb = 0.0, 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = mlm_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            counter.record(batch_tokens(batch), fwd_bwd=True, is_optimizer_step=True)
            tot += float(loss.item())
            nb += 1
        if epoch % max(1, eval_every) == 0 or epoch == epochs:
            rec = probe_now(epoch)
            rec["train_loss"] = tot / max(nb, 1)
            history.append(rec)

    save_model(model, str(arm.acquired_ckpt))
    del opt
    return {
        "checkpoint": str(arm.acquired_ckpt),
        "n_train": len(train_seqs), "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "seed": seed, "history": history, "flops": counter.as_dict(),
        "model": model,
    }


# --------------------------------------------------------------------------- #
# Weight-space: Delta = theta_1 - theta_0
# --------------------------------------------------------------------------- #
def matrix_deltas(model_a, model_b, *, encoder_only: bool = False) -> Dict[str, "object"]:
    """{param_name: (W_a - W_b)} over all 2-D weights, on CPU float32.

    2-D only: LayerNorms and biases carry no subspace structure to truncate, and
    including them would make "rank" meaningless. Tied weights appear once
    (``named_parameters`` de-duplicates), so an edit applies consistently.
    """
    import torch

    sb = dict(model_b.named_parameters())
    out = {}
    for name, pa in model_a.named_parameters():
        if pa.ndim != 2 or name not in sb:
            continue
        if encoder_only and ".encoder.layer." not in name:
            continue
        out[name] = (pa.detach().float().cpu() - sb[name].detach().float().cpu())
    return out


def rank_stats(sv) -> dict:
    """Rank descriptors for one matrix's singular-value spectrum.

    ``erank`` (Roy & Vetterli) = exp(entropy of the normalised spectrum); the
    participation ratio and stable rank are reported alongside because they weight the
    tail differently and a one-number claim about "high-rank" should not hinge on which
    definition was picked.
    """
    import torch

    sv = sv[sv > 0]
    if sv.numel() == 0:
        # An exactly-zero update (a weight the acquisition never touched). Zero, not NaN:
        # these are aggregated with a zero weight, and 0 * NaN would poison the mean.
        return {"erank": 0.0, "participation_ratio": 0.0, "stable_rank": 0.0,
                "full_rank": 0, "erank_frac": 0.0, "top1_energy_frac": 0.0,
                "top24_energy_frac": 0.0, "fro_norm": 0.0}
    p = sv / sv.sum()
    ent = float(-(p * p.clamp_min(1e-30).log()).sum())
    e2 = sv.pow(2)
    full = int(sv.numel())
    return {
        "erank": math.exp(ent),
        "erank_frac": math.exp(ent) / full,
        "participation_ratio": float(e2.sum() ** 2 / e2.pow(2).sum()),
        "stable_rank": float(e2.sum() / e2[0]),
        "full_rank": full,
        "top1_energy_frac": float(e2[0] / e2.sum()),
        "top24_energy_frac": float(e2[: min(24, full)].sum() / e2.sum()),
        "fro_norm": float(e2.sum().sqrt()),
    }


# --------------------------------------------------------------------------- #
# Subspace algebra — every basis is [k, d] with ORTHONORMAL ROWS in the same space
# --------------------------------------------------------------------------- #
def orth_rows(M, tol: float = 1e-6):
    """Orthonormalise the rows of ``M`` [k, d] (dropping numerically dependent rows)."""
    import torch

    if M.shape[0] == 0:
        return M
    Q, R = torch.linalg.qr(M.t().contiguous())
    keep = torch.abs(torch.diagonal(R)) > tol
    return Q[:, keep].t().contiguous()


def top_right_singular(delta, k: int):
    """Top-``k`` RIGHT singular vectors of ``delta`` [out, in] -> [k, in].

    For a weight matrix W: h_out = W h_in, so the right singular vectors are the
    *input-side* directions the update reads from — i.e. they live in the same
    down-projection input space as CIR's collapse basis and the loss gradient wrt the
    activations. That is what makes footprint / CIR-target / generator comparable.
    """
    import torch

    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    k = max(0, min(int(k), Vh.shape[0]))
    return Vh[:k].contiguous(), S


def overlap(X, Y) -> dict:
    """Overlap of two orthonormal row-bases in the same d-dim space.

    ``energy_X_in_Y`` = mean over X's directions of the squared norm of their projection
    onto span(Y) in [0,1]. Reported against the chance level for random subspaces of the
    same dimensions (k_Y/d) — in a 1920-dim space with k=24, chance is 1.25%, so a raw
    overlap must be read as a multiple of chance, never on its own.
    """
    import torch

    kx, d = X.shape
    ky = Y.shape[0]
    S = X @ Y.t()
    fro2 = float((S * S).sum())
    cos = torch.linalg.svdvals(S).clamp(0, 1)
    chance_x = ky / d
    chance_y = kx / d
    return {
        "k_X": int(kx), "k_Y": int(ky), "dim": int(d),
        "energy_X_in_Y": fro2 / max(kx, 1),
        "energy_Y_in_X": fro2 / max(ky, 1),
        "chance_X_in_Y": chance_x,
        "ratio_over_chance": (fro2 / max(kx, 1)) / chance_x if chance_x > 0 else float("nan"),
        "mean_cos_principal_angle": float(cos.mean()) if cos.numel() else float("nan"),
        "max_cos_principal_angle": float(cos.max()) if cos.numel() else float("nan"),
        "n_angles_ge_0.9": int((cos >= 0.9).sum()),
        "cos_principal_angles": [round(float(c), 4) for c in cos],
    }


def random_basis(dim: int, k: int, seed: int, device, dtype=None):
    """A Haar-random orthonormal k-frame — the negative control for every ablation."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn(dim, k, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(M)
    out = Q[:, :k].t().contiguous().to(device)
    return out if dtype is None else out.to(dtype)


# --------------------------------------------------------------------------- #
# Causal intervention: project a subspace out of the FFN down-proj input
# --------------------------------------------------------------------------- #
class ActProjection:
    """Context manager that removes span(basis_l) from ``output.dense``'s input at each
    targeted layer — the by-hand version of what CIR's collapse does, applied to a frozen
    model at inference time.

    Doing it as an activation projection rather than weight surgery is deliberate: it
    removes the subspace *exactly*, with no optimisation, no retain objective and no
    termination rule, so the answer "does F_test survive without these directions?" is
    about the subspace alone and not about any property of the CIR optimiser.
    """

    def __init__(self, model, bases: Dict[int, "object"]):
        self.model = model
        self.bases = bases
        self.handles: List = []

    def __enter__(self):
        import torch

        def mk(V):
            def pre_hook(_mod, args):
                x = args[0]
                Vd = V.to(device=x.device, dtype=x.dtype)
                return (x - (x @ Vd.t()) @ Vd,) + tuple(args[1:])
            return pre_hook

        for li, V in self.bases.items():
            if V is None or V.shape[0] == 0:
                continue
            self.handles.append(dense(self.model, int(li)).register_forward_pre_hook(mk(V)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


# --------------------------------------------------------------------------- #
# Activation / gradient second moments in the down-proj input space
# --------------------------------------------------------------------------- #
def act_grad_moments(model, tok, layers: Sequence[int], seqs: Sequence[str], *, device: str,
                     batch_size: int = 8, mlm_probability: float = 0.15,
                     max_length: int = MAX_LENGTH, seed: int = 0, max_seqs: int = 1024) -> dict:
    """Second moments of the MLM-loss gradient wrt the down-proj input activations.

    ``grad`` here is dL_MLM/da at ``output.dense``'s input, accumulated over all
    non-special token positions. Its top eigenvectors are this experiment's operational
    definition of the **generating directions** for whichever corpus is passed in: the
    directions along which the model's loss on that corpus is most sensitive. It is a
    first-order proxy, which is why t4 validates it causally (ablate it and the corpus
    PPPL must blow up) against a random-subspace control.

    Uses the UNCENTRED second moment E[g g^T] (not the covariance): the mean gradient is
    itself a real descent direction and dropping it would discard the single most
    important direction.
    """
    import torch

    from src.data.datamodule import make_mlm_loader, subsample
    from src.train.cir import _Cov, special_id_set

    seqs = subsample(list(seqs), max_seqs, seed=seed)
    special = special_id_set(tok)
    inter = dense(model, layers[0]).in_features
    acc_g = {li: _Cov(inter, device, dtype=torch.float32) for li in layers}
    acc_a = {li: _Cov(inter, device, dtype=torch.float32) for li in layers}

    cap: dict = {}
    handles = [dense(model, li).register_forward_hook(
        (lambda li_: (lambda _m, inp, _o: cap.__setitem__(li_, inp[0])))(li)) for li in layers]

    loader = make_mlm_loader(seqs, tok, batch_size, mlm_probability, max_length, shuffle=False, seed=seed)
    was_training = model.training
    model.eval()                       # no dropout; gradients still flow
    n_batches = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        m = attn.bool()
        for sid in special:
            m = m & (ids != sid)
        if m.sum() == 0:
            continue
        cap.clear()
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        acts = [cap[li] for li in layers]
        grads = torch.autograd.grad(out.loss, acts, retain_graph=False)
        for j, li in enumerate(layers):
            acc_g[li].add(grads[j].detach()[m])
            acc_a[li].add(acts[j].detach()[m])
        n_batches += 1
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    res = {}
    for li in layers:
        cg, mg = acc_g[li].cov_mean()
        ca, ma = acc_a[li].cov_mean()
        res[int(li)] = {
            "grad_second_moment": cg + torch.outer(mg, mg),   # E[g g^T]
            "act_cov": ca,
            "act_mean": ma,
            "n_tokens": acc_g[li].n,
        }
    print(f"  moments: {n_batches} batches, {res[int(layers[0])]['n_tokens']} tokens/layer", flush=True)
    return res


def top_eigvecs(M, k: int):
    """Top-``k`` eigenvectors (rows) of a symmetric PSD matrix."""
    from src.train.cir import _top_k_basis

    basis, evals = _top_k_basis(0.5 * (M + M.t()), None, k)
    return basis, evals


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def save_json(obj, path) -> Path:
    from src.utils.io import save_json as _sj

    _sj(_denan(obj), path)
    print(f"  wrote {path}", flush=True)
    return Path(path)


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


def _denan(o):
    if isinstance(o, dict):
        return {k: _denan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_denan(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def device_arg(prefer: str = "cuda") -> str:
    from src.utils.io import get_device

    return get_device(prefer)


def arm_args(parser):
    """Standard flags every phase script shares."""
    parser.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--seed", type=int, default=0)
    return parser
