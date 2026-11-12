"""Shared plumbing for experiment 5 — "readout or representation?".

THE CLAIM UNDER TEST
--------------------
The task-blocked model B produces an enormous *static* forget — 6,202x base F-PPPL on the
metagenomic anchor, up to 42,291x on PF00106_fullchain — that 16-64 real F examples undo,
giving a durability multiplier M ~= 1. The weight-space probe says B is a thin, in-basin
perturbation (rel ||dtheta|| B 2.17% vs control A' 5.00%; theta_0 -> theta_B monotone in
F-PPL, 0.30-nat barrier, flat benign path). So theta_0 and theta_B share a basis inside one
basin, and a *parameter splice* between them is a meaningful object rather than a category
error.

    Is the block a READOUT edit — something the last 0.69% of parameters (the MLM head)
    does to the logits — or a REPRESENTATIONAL edit that lives in the trunk?

Two hybrids answer it:

    H_head  = theta_0 trunk + B head   -> if this reproduces most of the block, readout.
    H_trunk = B trunk + theta_0 head   -> the complement.

Prior evidence is neutral-to-AGAINST the readout hypothesis, which is what makes this a
test rather than a formality: B's ``lm_head`` group moved 1.80% relative, i.e. LESS than
its 2.17% global displacement. Norm is not function; the functional splice is the
measurement.

WHAT THIS MODULE PROVIDES
-------------------------
* :data:`ARMS` — five existing pipeline run roots spanning a 210x -> 42,291x range of
  static block magnitude, so the headline number is a dose-response rather than one point.
  Nothing here re-derives data, tau, or checkpoints; every arm is read-only input.
* :class:`SpliceKit` — donor CPU state dicts + one scratch model, so an arm's hybrids are
  built by writing into a single resident model (the idiom ``lmc_path`` uses for LMC).
  **Hybrids are never written to disk**: ``save_pretrained`` would silently re-tie an
  untied H3 on reload, and the provenance we care about (which keys came from whom) is
  recorded in the JSON instead.
* :func:`frac_block` — the headline statistic, in log space. The block is 8.73 nats on the
  primary arm; ratios are the wrong scale to average or plot.
* :func:`partial_attack` — T3's freeze-then-fine-tune, replicating
  ``src/train/attack.py::run_attack``'s loop and ``best_target_ppl`` early-stopping
  semantics exactly, because T3's numbers are compared against existing sweep rows.

THE TIED-EMBEDDING TRAP (read before touching the splice)
---------------------------------------------------------
ESM-2 sets ``tie_word_embeddings: true``. ``esm.embeddings.word_embeddings.weight`` is
BOTH the input embedding (trunk) and the output projection (head), and
``lm_head.decoder.weight`` is the same storage under a second name. ``model.safetensors``
omits the duplicate (203 tensors); ``model.state_dict()`` exposes it. Consequences:

* Writing different values to the two names in one ``load_state_dict`` does NOT untie
  them — they share storage, last write wins.
* So "where the head ends" is genuinely ambiguous, and one arbitrary cut would be a
  confound. Hence the nested ladder H1 (headline) / H2 / H3, reported in full.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (plm_unlearning/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TASKBLOCK_ROOT", str(ROOT))

# A smoke/scratch run sets TB_RVR_TAG so its artifacts land beside — never on top of —
# the real ones.
TAG = os.environ.get("TB_RVR_TAG", "").strip()
_SUFFIX = f"_{TAG}" if TAG else ""
RESULTS = HERE / f"results{_SUFFIX}"

# The headline 35M model every arm was run at. theta_0 has no on-disk checkpoint — it is
# the HF id, which `load_checkpoint` accepts directly (see scripts/05_full_sweep.py:41).
MODEL_ID = "facebook/esm2_t12_35M_UR50D"
N_LAYERS = 12

# Scoring. Mirrors configs/config.yaml `eval:` EXACTLY — these are the settings that
# produced `forget_ppl_n0` in every run's summary.json (scripts/06_analyze.py:69-89), so
# any deviation would make our ratios incomparable to the 6,202x we are explaining.
N_EVAL = 256
EVAL_BATCH = 64
MASKS_PER_PASS = 8
MAX_LENGTH = 1024
SEED = 0

# T3 attacker recipe: read from the shared config so it stays in lockstep with the sweep.
ATTACK_CFG_PATH = ROOT / "configs" / "attack" / "full_ft.yaml"
EVAL_EVERY = 4                               # configs/config.yaml eval.eval_every
T3_BUDGETS = [16, 64, 256]
T3_SEEDS = [0, 1, 2]
T3_MODES = ["head_only", "trunk_only"]

# Head-only alpha path (T1) and depth ladder (T2).
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
K_LADDER = [0, 1, 2, 4, 8, 12]


# --------------------------------------------------------------------------- #
# Head sets — the nested ladder
# --------------------------------------------------------------------------- #
# Materialised against a real state_dict rather than hard-coded, so the partition is
# correct whether or not this transformers version registers `lm_head.decoder.bias`.
DECODER_KEY = "lm_head.decoder.weight"
WORD_EMB_KEY = "esm.embeddings.word_embeddings.weight"
FINAL_LN_PREFIX = "esm.encoder.emb_layer_norm_after."

HEAD_SET_ORDER = ["H1", "H2", "H3"]
HEADLINE_HEAD_SET = "H1"
HEAD_SET_DOC = {
    "H1": "lm_head.{dense,layer_norm,bias} only — decoder stays tied, so the output "
          "projection follows the TRUNK donor. The headline cut.",
    "H2": "H1 + esm.encoder.emb_layer_norm_after (the last op before the head).",
    "H3": "H2 + the decoder matrix, explicitly UNTIED: the trunk keeps its own input "
          "embedding while the head donor supplies the output projection.",
}


def head_keys(sd_keys: Sequence[str], head_set: str) -> List[str]:
    """Plain (non-tied) state-dict keys transferred from the head donor for ``head_set``.

    The decoder is deliberately excluded here for every set: under H1/H2 it stays tied to
    the trunk's word embedding, and under H3 it is handled by :func:`_untie_decoder`,
    which needs a parameter swap rather than a state-dict write.
    """
    if head_set not in HEAD_SET_ORDER:
        raise ValueError(f"unknown head set {head_set!r}; known: {HEAD_SET_ORDER}")
    keys = [k for k in sd_keys if k.startswith("lm_head.") and not k.startswith("lm_head.decoder")]
    if head_set in ("H2", "H3"):
        keys += [k for k in sd_keys if k.startswith(FINAL_LN_PREFIX)]
    return sorted(keys)


def layer_keys(sd_keys: Sequence[str], k: int) -> List[str]:
    """State-dict keys of the LAST ``k`` encoder layers (T2's depth ladder)."""
    want = {f"esm.encoder.layer.{i}." for i in range(N_LAYERS - k, N_LAYERS)}
    return sorted(key for key in sd_keys if any(key.startswith(p) for p in want))


# --------------------------------------------------------------------------- #
# Arms — five existing run roots, read-only
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    key: str
    label: str
    family: str
    run_root: Path
    note: str = ""

    @property
    def data(self) -> Path:
        return self.run_root / "data"

    @property
    def checkpoints(self) -> Path:
        return self.run_root / "checkpoints"

    @property
    def B(self) -> Path:
        return self.checkpoints / "B"

    @property
    def A_prime(self) -> Path:
        return self.checkpoints / "A_prime"

    @property
    def results(self) -> Path:
        d = RESULTS / self.key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def summary(self) -> dict:
        p = self.run_root / "results" / "summary.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def thresholds(self) -> dict:
        p = self.run_root / "results" / "thresholds.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def weightspace(self) -> dict:
        p = self.run_root / "results" / "weightspace.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def expected(self) -> Dict[str, Optional[float]]:
        """The pipeline's own n=0 F-PPPL per checkpoint — T0's reproduction target."""
        fn0 = self.summary().get("forget_ppl_n0", {}) or {}
        return {"theta0": fn0.get("A"), "B": fn0.get("B"), "A_prime": fn0.get("A_prime")}

    def lm_head_rel_l2(self) -> Dict[str, Optional[float]]:
        """B's and A''s per-group relative ||dtheta|| for `lm_head` and overall.

        Reported beside frac_block: a large FUNCTIONAL fraction carried by a group whose
        norm displacement is BELOW the model average is the sharpest form of the finding.
        """
        l2 = (self.weightspace().get("l2") or {})
        out: Dict[str, Optional[float]] = {}
        for name in ("B", "A_prime"):
            g = l2.get(name) or {}
            out[f"{name}_lm_head_rel"] = (g.get("lm_head") or {}).get("rel")
            out[f"{name}_total_rel"] = (g.get("__total__") or {}).get("rel")
        return out


ARMS: Dict[str, Arm] = {
    # Primary: the only substrate in the project with genuine acquisition headroom
    # (29.5%), where M ~= 0.96-1.01 is clean and uncensored. Static block 6,202x.
    "mgnify": Arm(
        key="mgnify",
        label="MGnify MGYF0000018690 — metagenomic (primary, 6,202x)",
        family="MGYF0000018690",
        run_root=ROOT / "runs" / "MGYF18690_metagenomic",
        note="Source-aware split; 0 F_test/F_train hits at -c0.8. The headline arm.",
    ),
    # The largest static block in the project.
    "pfam_fullchain": Arm(
        key="pfam_fullchain",
        label="Pfam PF00106 full-chain (42,291x)",
        family="PF00106",
        run_root=ROOT / "runs" / "PF00106_fullchain",
    ),
    "pfam_legacy": Arm(
        key="pfam_legacy",
        label="Pfam PF00106 legacy root (2,989x)",
        family="PF00106",
        run_root=ROOT,
        note="The original run; its F_test is train-homologous, which inflates the "
             "static block. Harmless here — this experiment only decomposes whatever "
             "block exists, it does not interpret its size.",
    ),
    "pfam_domain": Arm(
        key="pfam_domain",
        label="Pfam PF00106 domain-clipped (1,371x)",
        family="PF00106",
        run_root=ROOT / "runs" / "PF00106_domain",
        note="Headroom collapsed here, so M is unreliable — but the STATIC block is "
             "well-defined and that is all this experiment decomposes.",
    ),
    "pf00107": Arm(
        key="pf00107",
        label="Pfam PF00107 full-chain (210x)",
        family="PF00107",
        run_root=ROOT / "runs" / "PF00107_fullchain",
    ),
}

ARM_ORDER = ["mgnify", "pfam_fullchain", "pfam_legacy", "pfam_domain", "pf00107"]
PRIMARY_ARM = "mgnify"


def get_arm(key: str) -> Arm:
    if key not in ARMS:
        raise SystemExit(f"unknown arm {key!r}; known: {', '.join(ARM_ORDER)}")
    return ARMS[key]


# --------------------------------------------------------------------------- #
# Evaluation corpora
# --------------------------------------------------------------------------- #
def data_paths(arm: Arm):
    from src.data.datamodule import DataPaths

    return DataPaths(str(arm.data))


def eval_corpora(arm: Arm, n_eval: int = N_EVAL, seed: int = SEED) -> Dict[str, List[str]]:
    """The two scoring corpora, built EXACTLY as scripts/06_analyze.py:69-73 builds them.

    ``F``      — the held-out forget split, the set the 6,202x is measured on.
    ``benign`` — the retain holdout. Not a footnote: a hybrid with a huge F-PPPL AND a
                 wrecked benign PPPL is a broken model, not a readout block, and that is
                 this experiment's falsifier.
    """
    from src.data.datamodule import subsample, target_test_seqs
    from src.utils.io import fasta_seqs

    paths = data_paths(arm)
    if not Path(paths.F_test).exists():
        raise SystemExit(f"missing {paths.F_test} for arm {arm.key} — is the run root correct?")
    return {
        "F": subsample(target_test_seqs(paths, "F"), n_eval, seed=seed),
        "benign": subsample(fasta_seqs(paths.retain_holdout), n_eval, seed=seed),
    }


def tokenizer():
    from src.models.load import load_tokenizer

    return load_tokenizer(MODEL_ID)


def ppl(model, tok, seqs: Sequence[str], device: str,
        masks_per_pass: int = MASKS_PER_PASS, batch_size: int = EVAL_BATCH) -> float:
    from src.eval.pseudo_perplexity import pseudo_perplexity

    if not seqs:
        return float("nan")
    return pseudo_perplexity(model, tok, seqs, device, batch_size, MAX_LENGTH,
                             masks_per_pass=masks_per_pass).ppl


def score(model, tok, corpora: Dict[str, Sequence[str]], device: str,
          masks_per_pass: int = MASKS_PER_PASS, label: str = "") -> Dict[str, float]:
    out = {name: ppl(model, tok, seqs, device, masks_per_pass) for name, seqs in corpora.items()}
    print(f"  {label:<28} F={out.get('F', float('nan')):>12.4f}   "
          f"benign={out.get('benign', float('nan')):>8.4f}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# The headline statistic
# --------------------------------------------------------------------------- #
def frac_block(ppl_m: float, ppl_base: float, ppl_B: float) -> Optional[float]:
    """Fraction of the block reproduced by ``m``, in LOG space.

        frac = (log ppl_m - log ppl_base) / (log ppl_B - log ppl_base)

    0 at theta_0 and 1 at B by construction. Log space because the block is ~8.7 nats on
    the primary arm: a ratio-space average would be dominated entirely by its endpoint.
    Returns None (never inf) when an arm's B did not block, so a degenerate arm drops out
    of the aggregate instead of poisoning it.
    """
    for v in (ppl_m, ppl_base, ppl_B):
        if v is None or not math.isfinite(v) or v <= 0:
            return None
    denom = math.log(ppl_B) - math.log(ppl_base)
    if abs(denom) < 1e-9:
        return None
    return (math.log(ppl_m) - math.log(ppl_base)) / denom


def benign_drift(ppl_m: float, ppl_base: float) -> Optional[float]:
    if not (ppl_m and ppl_base) or not math.isfinite(ppl_m) or not math.isfinite(ppl_base):
        return None
    return ppl_m / ppl_base - 1.0


# --------------------------------------------------------------------------- #
# The splice
# --------------------------------------------------------------------------- #
def splice_state_dicts(sd_trunk: Dict, sd_head: Dict, transferred: Sequence[str],
                       untied: bool) -> Dict:
    """``sd_trunk`` everywhere except ``transferred``, which comes from ``sd_head``.

    Pure and model-free, so the partition can be unit-tested without a checkpoint. The
    decoder is set explicitly rather than left to chance, because it is the one key whose
    correct source depends on whether the caller intends to untie:

    * tied (H1/H2)  -> the decoder mirrors the TRUNK's word embedding, which is what
      leaving ``tie_word_embeddings`` alone actually does at inference.
    * untied (H3)   -> the decoder carries the HEAD donor's word embedding while the trunk
      keeps its own input embedding.
    """
    out = {k: v for k, v in sd_trunk.items()}
    for k in transferred:
        if k not in sd_head:
            raise KeyError(f"head donor is missing {k!r}")
        out[k] = sd_head[k]
    if DECODER_KEY in out:
        out[DECODER_KEY] = (sd_head if untied else out)[WORD_EMB_KEY]
    return out


def _untie_decoder(model, decoder_weight) -> None:
    """Give ``lm_head.decoder`` its own storage, loaded with ``decoder_weight``.

    Setting the two tied names to different values through ``load_state_dict`` would NOT
    untie them (shared storage, last write wins), so the parameter object itself has to be
    replaced. ``config.tie_word_embeddings`` is flipped so nothing downstream re-ties, and
    the caller asserts the storages really did separate.
    """
    import torch

    model.config.tie_word_embeddings = False
    model.lm_head.decoder.weight = torch.nn.Parameter(decoder_weight.clone().to(
        model.lm_head.decoder.weight.device, model.lm_head.decoder.weight.dtype))
    if model.lm_head.decoder.weight.data_ptr() == model.esm.embeddings.word_embeddings.weight.data_ptr():
        raise RuntimeError("untie failed: decoder still shares storage with word_embeddings")


class SpliceKit:
    """Donor state dicts for one arm plus a single scratch model the hybrids are built in.

    Loading three 35M checkpoints per hybrid would dominate the runtime; instead the
    donors are held as CPU state dicts and each hybrid is materialised by writing into one
    resident GPU model — the idiom ``src/analysis/weightspace.py::lmc_path`` uses to walk
    the LMC path. The scratch model is rebuilt only when crossing the tied/untied
    boundary, which happens at most twice per arm.
    """

    def __init__(self, arm: Arm, device: str, donors: Optional[Dict[str, str]] = None):
        from src.models.load import load_checkpoint

        self.arm = arm
        self.device = device
        self.donor_paths: Dict[str, str] = donors or {
            "theta0": MODEL_ID,
            "B": str(arm.B),
            "A_prime": str(arm.A_prime),
        }
        for name, path in self.donor_paths.items():
            if path != MODEL_ID and not (Path(path) / "config.json").exists():
                raise SystemExit(
                    f"missing checkpoint {name} for arm {arm.key} at {path} — "
                    f"this experiment consumes the main pipeline's outputs; run that first")
        self.sd: Dict[str, Dict] = {}
        for name, path in self.donor_paths.items():
            m = load_checkpoint(path, device="cpu")
            self.sd[name] = {k: v.detach().clone() for k, v in m.state_dict().items()}
            del m
        self.keys = sorted(self.sd["theta0"].keys())
        self._model = None
        self._untied = None
        self._scratch()

    # -- scratch model management ------------------------------------------- #
    def _scratch(self, untied: bool = False):
        from src.models.load import load_checkpoint

        if self._model is not None and self._untied == untied:
            return self._model
        self._model = None
        self._free()
        self._model = load_checkpoint(MODEL_ID, device=self.device)
        self._untied = untied
        if untied:
            _untie_decoder(self._model, self.sd["theta0"][DECODER_KEY])
        elif self._model.lm_head.decoder.weight.data_ptr() != \
                self._model.esm.embeddings.word_embeddings.weight.data_ptr():
            raise RuntimeError("expected tied decoder on a freshly loaded ESM-2 checkpoint")
        return self._model

    def _free(self):
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        self._model = None
        self.sd = {}
        self._free()

    # -- the splice itself --------------------------------------------------- #
    def build(self, trunk: str, head: str, head_set: str = HEADLINE_HEAD_SET,
              extra_keys: Sequence[str] = ()) -> Tuple[object, dict]:
        """``trunk`` donor's weights everywhere except ``head_set`` (+ ``extra_keys``).

        ``extra_keys`` is how T2 walks the depth ladder — H1 plus the last k encoder
        layers — without needing a second code path.
        """
        transferred = sorted(set(head_keys(self.keys, head_set)) | set(extra_keys))
        untied = head_set == "H3"
        model = self._scratch(untied=untied)

        # Under H1/H2 the tied double-write is a no-op (both names carry the trunk's
        # value); under H3 the decoder carries the head donor's embedding instead.
        sd = splice_state_dicts(self.sd[trunk], self.sd[head], transferred, untied)
        model.load_state_dict(sd, strict=True)
        model.eval()

        n_params = sum(int(self.sd[trunk][k].numel()) for k in transferred)
        if untied:
            n_params += int(self.sd[trunk][WORD_EMB_KEY].numel())
        prov = {
            "trunk": trunk, "head": head, "head_set": head_set,
            "trunk_ckpt": self.donor_paths[trunk], "head_ckpt": self.donor_paths[head],
            "keys_from_head": transferred,
            "decoder_untied": untied,
            "n_keys_from_head": len(transferred) + (1 if untied else 0),
            "n_params_from_head": n_params,
            "n_params_total": self.n_params_total,
            "frac_params_from_head": n_params / self.n_params_total,
        }
        return model, prov

    def build_alpha(self, trunk: str, head: str, alpha: float,
                    head_set: str = HEADLINE_HEAD_SET) -> Tuple[object, dict]:
        """``trunk`` everywhere, with the head set linearly interpolated trunk -> head.

        Reuses ``interpolate_state_dicts``' convention restricted to the head keys, so the
        alpha path is directly comparable to the project's existing LMC figures.
        """
        transferred = head_keys(self.keys, head_set)
        if head_set == "H3":
            raise ValueError("alpha path is defined on the tied head sets only (H1/H2)")
        model = self._scratch(untied=False)
        sd = {k: v for k, v in self.sd[trunk].items()}
        for k in transferred:
            a, b = self.sd[trunk][k].float(), self.sd[head][k].float()
            sd[k] = ((1.0 - alpha) * a + alpha * b).to(self.sd[trunk][k].dtype)
        sd[DECODER_KEY] = sd[WORD_EMB_KEY]
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, {"trunk": trunk, "head": head, "head_set": head_set, "alpha": float(alpha)}

    def donor(self, name: str):
        """The unspliced donor itself, materialised in the scratch model."""
        model = self._scratch(untied=False)
        sd = {k: v for k, v in self.sd[name].items()}
        sd[DECODER_KEY] = sd[WORD_EMB_KEY]
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

    @property
    def n_params_total(self) -> int:
        """Unique parameter count — the tied decoder is counted once, not twice."""
        return sum(int(v.numel()) for k, v in self.sd["theta0"].items() if k != DECODER_KEY)

    def head_param_counts(self) -> Dict[str, dict]:
        tot = self.n_params_total
        out = {}
        for hs in HEAD_SET_ORDER:
            n = sum(int(self.sd["theta0"][k].numel()) for k in head_keys(self.keys, hs))
            if hs == "H3":
                n += int(self.sd["theta0"][WORD_EMB_KEY].numel())
            out[hs] = {"n_params": n, "frac_of_model": n / tot, "doc": HEAD_SET_DOC[hs]}
        return out


# --------------------------------------------------------------------------- #
# T3 — freeze, then fine-tune
# --------------------------------------------------------------------------- #
def attack_cfg() -> dict:
    """The sweep's own full-FT attacker recipe, read (never edited) from configs/."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ATTACK_CFG_PATH)
    return {k: cfg.get(k) for k in
            ("mode", "lr", "max_epochs", "batch_size", "mlm_probability", "weight_decay", "patience")}


def trainable_mask(model, mode: str) -> Tuple[int, int]:
    """Freeze everything outside ``mode``'s parameter set. Returns (n_trainable, n_frozen).

    ``named_parameters`` de-duplicates tied weights, so ``word_embeddings`` appears once
    and the decoder inherits its requires_grad. That is the intended semantics: under
    ``head_only`` the output projection is TRUNK-side and stays frozen, matching H1.
    """
    keys = set(head_keys([n for n, _ in model.named_parameters()], HEADLINE_HEAD_SET))
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        want = (name in keys) if mode == "head_only" else (name not in keys)
        p.requires_grad_(want)
        if want:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    return n_train, n_frozen


def partial_attack(checkpoint_path: str, tok, attacker_seqs: Sequence[str],
                   target_seqs: Sequence[str], mode: str, device: str, *,
                   seed: int = 0, cfg: Optional[dict] = None,
                   eval_every: int = EVAL_EVERY, max_eval_seqs: int = N_EVAL,
                   masks_per_pass: int = MASKS_PER_PASS,
                   max_length: int = MAX_LENGTH) -> dict:
    """Fine-tune only ``mode``'s parameters of ``checkpoint_path`` toward F.

    Deliberately mirrors ``src/train/attack.py::run_attack`` step for step — the same
    loader, loss, clipping, ``eval_every`` cadence, ``best_target_ppl`` tracking and
    patience rule, and the same ``subsample(target, max_eval_seqs, seed=seed)`` scoring
    set. That equivalence is load-bearing: T3's numbers are read against existing sweep
    rows, which report ``best_target_ppl`` under exactly this protocol. ``run_attack``
    itself cannot be reused because it loads the checkpoint internally and takes
    ``[p for p in model.parameters() if p.requires_grad]`` with no hook for freezing, and
    editing src/ is out of scope for this experiment.
    """
    import torch

    from src.data.datamodule import make_mlm_loader, subsample
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.models.load import load_checkpoint
    from src.train.mlm import batch_tokens, mlm_loss
    from src.utils.flops import FlopCounter
    from src.utils.seeding import set_seed

    c = cfg or attack_cfg()
    set_seed(seed)
    target_seqs = subsample(list(target_seqs), max_eval_seqs, seed=seed)

    model = load_checkpoint(checkpoint_path, device=device)
    n_train_params, n_frozen = trainable_mask(model, mode)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(c["lr"]), weight_decay=float(c["weight_decay"]))
    counter = FlopCounter(n_params=n_train_params)
    loader = make_mlm_loader(attacker_seqs, tok, int(c["batch_size"]), float(c["mlm_probability"]),
                             max_length, shuffle=True, seed=seed)

    def eval_target() -> float:
        return pseudo_perplexity(model, tok, target_seqs, device, EVAL_BATCH,
                                 max_length, masks_per_pass).ppl

    history = [{"epoch": 0, "target_ppl": eval_target()}]
    best_ppl, best_epoch, bad = history[0]["target_ppl"], 0, 0
    max_epochs, patience = int(c["max_epochs"]), int(c["patience"])
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = mlm_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            counter.record(batch_tokens(batch), fwd_bwd=True, is_optimizer_step=True)
        if not (epoch % max(1, eval_every) == 0 or epoch == max_epochs):
            continue
        p = eval_target()
        history.append({"epoch": epoch, "target_ppl": p})
        if p < best_ppl - 1e-6:
            best_ppl, best_epoch, bad = p, epoch, 0
        else:
            bad += 1
            if patience and bad >= patience:
                break

    res = {
        "best_target_ppl": best_ppl,
        "final_target_ppl": history[-1]["target_ppl"],
        "naive_target_ppl": history[0]["target_ppl"],
        "best_epoch": best_epoch,
        "epochs_run": len(history) - 1,
        "n_train": len(attacker_seqs),
        "trainable_params": n_train_params,
        "frozen_params": n_frozen,
        "flops": counter.as_dict(),
        "history": history,
    }
    del model, opt, params, loader
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def sweep_baseline(arm: Arm, budgets: Sequence[int] = tuple(T3_BUDGETS),
                   recipe: str = "full_ft") -> Dict[str, Optional[float]]:
    """B's existing full-FT relearning curve at the same budgets — free, from disk.

    The main sweep already measured this under an identical protocol, so T3 needs no new
    compute for its baseline. Returns median ``best_target_ppl`` over seeds per budget.
    """
    import glob
    import statistics

    rows: Dict[int, List[float]] = {int(b): [] for b in budgets}
    for path in sorted(glob.glob(str(arm.run_root / "results" / "sweep" / "*shard*.csv"))):
        try:
            import csv

            with open(path) as fh:
                for r in csv.DictReader(fh):
                    if (r.get("model") == "B" and r.get("recipe") == recipe
                            and r.get("adversary_data") == "clean"
                            and r.get("target_family") == "F"):
                        b = int(float(r["budget_n"]))
                        if b in rows:
                            rows[b].append(float(r["best_target_ppl"]))
        except Exception as e:  # noqa: BLE001 — a malformed shard must not kill the phase
            print(f"  [baseline] skipped {Path(path).name}: {e!r}", flush=True)
    return {str(b): (statistics.median(v) if v else None) for b, v in rows.items()}


# --------------------------------------------------------------------------- #
# IO / CLI
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


def device_arg(prefer: str = "cuda", allow_cpu_fallback: bool = False) -> str:
    """Resolve the device, and REFUSE to silently fall back to CPU.

    ``src.utils.io.get_device`` never hard-fails: if CUDA is missing it quietly returns
    "cpu". That is the right default for the main pipeline, but it is dangerous here — a
    transient CUDA-init fault on an allocated GPU node once turned a T3 run into an
    18-hour CPU crawl that hit its wall having written no artifact at all. A slow success
    that produces nothing is far worse than an immediate, legible failure, so an explicit
    ``--device cuda`` that cannot be honoured is now an error.

    Pass ``--device cpu`` (or ``--allow-cpu-fallback``) when CPU is genuinely intended.
    """
    from src.utils.io import get_device

    resolved = get_device(prefer)
    if prefer.startswith("cuda") and not resolved.startswith("cuda") and not allow_cpu_fallback:
        import os

        raise SystemExit(
            "--device cuda was requested but torch.cuda.is_available() is False.\n"
            f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '?')} "
            f"node={os.environ.get('SLURMD_NODENAME', '?')} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
            "  The GPU may have been allocated but failed to initialise (a node fault) — "
            "resubmit, and exclude that node if it recurs.\n"
            "  To run on CPU deliberately, pass --device cpu or --allow-cpu-fallback."
        )
    return resolved


def arm_args(parser):
    """Standard flags every phase script shares."""
    parser.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--masks-per-pass", type=int, default=MASKS_PER_PASS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="permit silently running on CPU when --device cuda cannot be "
                             "honoured (default: hard-fail, see device_arg)")
    return parser


def protocol_block(n_eval: int, masks_per_pass: int, seed: int) -> dict:
    """Echoed into every artifact so each JSON is self-contained."""
    return {
        "model_id": MODEL_ID, "n_eval": n_eval, "eval_batch": EVAL_BATCH,
        "masks_per_pass": masks_per_pass, "max_length": MAX_LENGTH, "seed": seed,
        "matches": "configs/config.yaml eval: + scripts/06_analyze.py:69-89",
    }
