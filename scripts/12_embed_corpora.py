"""Mean-pooled ESM-2 embeddings for F / F' / ladder / retain, cached to .npy.

Motivation. The project reasons constantly about "distance from F" — ladder rungs are
picked by HMM identity bands (close / mid / far / background) and every headline result
(sibling-ladder leakage, footprint-vs-generator, readout-vs-representation) turns on how
close a corpus sits to the forget family. That distance has only ever been measured in
SEQUENCE space (mmseqs / HMM identity). This script produces the inputs needed to look at
the same corpora in the model's REPRESENTATION space, which is an independent check on
several of those claims — e.g. MGnify rung 6 (MGYF0000026700) is labelled
`band: background` but carries mean_id_all 0.25, higher than rung 3.

Stage 1 of 2. This is the GPU half: embed every corpus once and cache it. Stage 2 is
`scripts/13_umap_plot.py`, which fits the UMAP and draws the figure. They are split so
that re-tuning n_neighbors / min_dist does not mean re-embedding 240k sequences.

Pooling is `src.eval.membership.embed_sequences` verbatim (the project's one mean-pooler;
it already masks specials/pad), wrapped here with length-sorted bucketing — that function
pads per batch, so unsorted input would waste a large fraction of the 55M-token retain
pass on padding. Nothing under src/ is modified.

    python scripts/12_embed_corpora.py --smoke          # 200 seqs/group, quick shape check
    python scripts/12_embed_corpora.py                  # full run (~72M tokens)
    python scripts/12_embed_corpora.py --anchors mgnify # one anchor only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import _bootstrap  # noqa: F401  (puts ROOT on sys.path)

import numpy as np

ROOT = _bootstrap.ROOT

# Held-out/train splits of the SAME family must land on top of each other; they are kept
# as separate groups precisely so that can be checked (see the sanity checks in 13_).
ROLE_F_TRAIN = "F_train"
ROLE_F_TEST = "F_test"
ROLE_F_PRIME = "F_prime"
ROLE_RUNG = "rung"
ROLE_RETAIN = "retain"


@dataclass
class Group:
    """One corpus to embed. `homology` is the ladder's own identity-to-F number
    (mean_id_all for Pfam, hh_identity_pct/100 for MGnify); None where not applicable.
    `ckpt` is the model this group is embedded with — one per anchor for variant B."""

    anchor: str
    label: str
    role: str
    path: str
    homology: Optional[float] = None
    band: Optional[str] = None
    rung: Optional[int] = None
    cap: Optional[int] = None  # None => use the global --max-cap
    ckpt: Optional[str] = None  # None => the --model default


# --------------------------------------------------------------------------- #
# Group table — built from each anchor's ladder_selected.json so that rung order
# and homology come from the source of truth rather than being retyped here.
# --------------------------------------------------------------------------- #
# Shared across all three anchors: the UniRef50 retain corpus. Embedded ONCE.
RETAIN_FASTA = "data/splits/retain_train.fasta"

# The active MGnify ladder is the DECONTAMINATED one. data/MGnify/pair1_ladder/ is the
# superseded v1 whose rungs were 21-86% literal F_train regions — different accessions
# (34691, 26909, 22633). Do not point this at that directory.
MGNIFY_ROOT = "runs/MGYF18690_ladder_decon"
MGNIFY_LADDER = f"{MGNIFY_ROOT}/data/ladder/MGYF0000018690"


def _mgnify_groups() -> List[Group]:
    sel = json.load(open(os.path.join(ROOT, MGNIFY_LADDER, "ladder_selected.json")))
    g = [
        Group("mgnify", "F_train", ROLE_F_TRAIN, f"{MGNIFY_ROOT}/data/splits/F_train.fasta"),
        Group("mgnify", "F_test", ROLE_F_TEST, f"{MGNIFY_ROOT}/data/splits/F_test.fasta"),
        # F_prime_test holds only 57 sequences after cross-filtering, too few to read on a
        # UMAP; the train split is the usable view of F'.
        Group("mgnify", "F_prime", ROLE_F_PRIME, f"{MGNIFY_ROOT}/data/splits/F_prime_train.fasta"),
    ]
    for r in sel["rungs"]:
        g.append(
            Group(
                anchor="mgnify",
                label=f"L{r['rung']}_{r['acc']}",
                role=ROLE_RUNG,
                path=f"{MGNIFY_LADDER}/corpora/{r['acc']}_ft.fasta",
                homology=float(r["mean_id_all"]),
                band=r["band"],
                rung=int(r["rung"]),
            )
        )
    return g


def _pfam_groups(anchor: str, f_prime_acc: str) -> List[Group]:
    base = f"data/ladder/{anchor}"
    sel = json.load(open(os.path.join(ROOT, base, "ladder_selected.json")))
    g = [
        Group(anchor, "F_train", ROLE_F_TRAIN, f"{base}/fullchain/F_train_full.fasta"),
        Group(anchor, "F_test", ROLE_F_TEST, f"{base}/fullchain/F_test_full.fasta"),
        Group(anchor, "F_prime", ROLE_F_PRIME, f"{base}/fullchain/{f_prime_acc}_train_full.fasta"),
    ]
    for r in sel["rungs"]:
        g.append(
            Group(
                anchor=anchor,
                label=f"L{r['rung']}_{r['acc']}",
                role=ROLE_RUNG,
                path=f"{base}/fullchain/{r['acc']}_ft_full.fasta",
                homology=float(r["mean_id_all"]),
                band=r["band"],
                rung=int(r["rung"]),
            )
        )
    return g


# The task-blocked model is trained PER ANCHOR, so there is no single "B" — each
# anchor's corpora (retain included) must be embedded with its own B. The two MGnify
# roots hold byte-identical B checkpoints (same md5), so either path works.
B_CKPT = {
    "mgnify": "runs/MGYF18690_ladder_decon/checkpoints/B",
    "PF00106": "runs/PF00106_fullchain/checkpoints/B",
    "PF00107": "runs/PF00107_fullchain/checkpoints/B",
}
A_PRIME_CKPT = {
    "mgnify": "runs/MGYF18690_ladder_decon/checkpoints/A_prime",
    "PF00106": "runs/PF00106_fullchain/checkpoints/A_prime",
    "PF00107": "runs/PF00107_fullchain/checkpoints/A_prime",
}
VARIANT_CKPT = {"base": None, "B": B_CKPT, "A_prime": A_PRIME_CKPT}


def build_groups(anchors: List[str], variant: str = "base") -> List[Group]:
    groups: List[Group] = []
    if "retain" in anchors and variant == "base":
        # cap=None + --max-cap ignored for retain: the whole corpus is wanted as the
        # ambient density the families are plotted against. Under `base` every anchor
        # shares one model, so retain is embedded once.
        groups.append(Group("shared", "retain", ROLE_RETAIN, RETAIN_FASTA, cap=-1))
    if "mgnify" in anchors:
        groups += _mgnify_groups()
    if "PF00106" in anchors:
        groups += _pfam_groups("PF00106", "PF01593")
    if "PF00107" in anchors:
        groups += _pfam_groups("PF00107", "PF01370")

    if variant != "base":
        table = VARIANT_CKPT[variant]
        # Per-anchor model => retain is no longer shared and is embedded once per anchor.
        per_anchor_retain = [
            Group(a, "retain", ROLE_RETAIN, RETAIN_FASTA, cap=-1)
            for a in dict.fromkeys(g.anchor for g in groups) if a in table
        ]
        groups += per_anchor_retain
        for g in groups:
            g.ckpt = table.get(g.anchor)
        groups = [g for g in groups if g.ckpt is not None]
    return groups


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def resolve_device(prefer: str, allow_cpu_fallback: bool) -> str:
    """Resolve the device and REFUSE to fall back to CPU silently.

    `src.utils.io.get_device` quietly returns "cpu" when CUDA is missing. That is right
    for the main pipeline but dangerous here: a transient CUDA-init fault on an allocated
    node would turn this into a multi-day CPU crawl that hits its wall having written
    nothing. Same rationale as `experiments/5_readout_vs_representation/common.py::device_arg`
    (not imported directly — that directory is not an importable package name).
    """
    from src.utils.io import get_device

    resolved = get_device(prefer)
    if prefer.startswith("cuda") and not resolved.startswith("cuda") and not allow_cpu_fallback:
        raise SystemExit(
            f"--device {prefer} requested but torch.cuda.is_available() is False.\n"
            f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '?')}  "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}\n"
            "  Pass --device cpu or --allow-cpu-fallback if CPU is genuinely intended."
        )
    return resolved


def embed_length_sorted(model, tokenizer, seqs: List[str], device: str, batch_size: int,
                        max_length: int) -> np.ndarray:
    """`embed_sequences` with length bucketing, then the permutation undone.

    embed_sequences tokenizes with padding=True, i.e. pads to the longest member of each
    batch. Retain has mean length 303 but a max of 14,607, so on unsorted input most
    batches pad far beyond their median. Sorting by length first keeps each batch roughly
    uniform; the inverse permutation restores the caller's order so rows still line up
    with the id list.
    """
    import torch

    from src.eval.membership import embed_sequences

    order = np.argsort([len(s) for s in seqs], kind="stable")
    ordered = [seqs[i] for i in order]

    if device.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.float16):
            feats = embed_sequences(model, tokenizer, ordered, device=device,
                                    batch_size=batch_size, max_length=max_length)
    else:
        feats = embed_sequences(model, tokenizer, ordered, device=device,
                                batch_size=batch_size, max_length=max_length)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(feats)[inverse]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D",
                    help="HF id; 35M is the project's headline model (embed_dim 480).")
    ap.add_argument("--anchors", default="retain,mgnify,PF00106,PF00107",
                    help="Comma-separated subset of: retain,mgnify,PF00106,PF00107")
    ap.add_argument("--out", default="results/umap/embeddings")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=1024,
                    help="Matches the project-wide eval max_length.")
    ap.add_argument("--max-cap", type=int, default=3000,
                    help="Per-family subsample cap (plot legibility, not compute). "
                         "Retain is exempt and always used in full.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="base", choices=sorted(VARIANT_CKPT),
                    help="base = the pretrained model given by --model; B / A_prime = "
                         "each anchor's own trained checkpoint (retain re-embedded per "
                         "anchor, since the model then differs between anchors).")
    ap.add_argument("--smoke", action="store_true",
                    help="200 seqs/group incl. retain — shape/throughput check only.")
    ap.add_argument("--force", action="store_true", help="Re-embed groups already cached.")
    args = ap.parse_args()

    from src.data.datamodule import subsample
    from src.models.load import load_mlm, load_tokenizer
    from src.utils.io import read_fasta

    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    groups = build_groups(anchors, variant=args.variant)
    device = resolve_device(args.device, args.allow_cpu_fallback)

    _bootstrap.banner(f"12_embed_corpora | variant={args.variant} | model={args.model} | "
                      f"device={device} | {len(groups)} groups | smoke={args.smoke}")

    out_root = os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)

    # Skip the model load entirely if every group is already cached.
    pending = []
    for g in groups:
        npy = os.path.join(out_root, g.anchor, f"{g.label}.npy")
        if os.path.exists(npy) and not args.force:
            print(f"  cached, skipping: {g.anchor}/{g.label}")
            continue
        pending.append(g)
    if not pending:
        print("nothing to do — all groups cached (use --force to re-embed)")
        return

    # The tokenizer is the pretrained ESM-2 one in every case: B and A' are finetunes,
    # so the vocabulary is unchanged and only the weights differ.
    tokenizer = load_tokenizer(args.model)

    manifest: Dict[str, dict] = {}
    manifest_path = os.path.join(out_root, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))

    # Load one model per distinct checkpoint, and order the work so each is loaded once.
    pending.sort(key=lambda g: (g.ckpt or "", g.anchor, g.label))
    model = None
    loaded_ckpt = "\0"  # sentinel: no checkpoint loaded yet

    for g in pending:
        want = g.ckpt or args.model
        if want != loaded_ckpt:
            if model is not None:
                del model
                import torch

                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            src_path = want if g.ckpt is None else os.path.join(ROOT, want)
            print(f"\nloading model: {want}")
            model = load_mlm(src_path, device=device)
            model.eval()
            loaded_ckpt = want
        fasta = os.path.join(ROOT, g.path)
        if not os.path.exists(fasta):
            raise SystemExit(f"missing FASTA for {g.anchor}/{g.label}: {fasta}")

        records = read_fasta(fasta)
        seqs = [r.seq for r in records]
        ids = [r.id for r in records]

        # cap == -1 marks retain: the whole corpus is the point.
        cap = 200 if args.smoke else (None if g.cap == -1 else args.max_cap)
        if cap is not None and len(seqs) > cap:
            # subsample() shuffles then slices, so sample indices the same way to keep
            # ids aligned with rows rather than calling it on the sequences directly.
            keep = subsample(list(range(len(seqs))), cap, seed=args.seed)
            keep = sorted(keep)
            seqs = [seqs[i] for i in keep]
            ids = [ids[i] for i in keep]

        t0 = time.time()
        feats = embed_length_sorted(model, tokenizer, seqs, device,
                                    args.batch_size, args.max_length)
        dt = time.time() - t0
        rate = len(seqs) / dt if dt > 0 else float("nan")

        anchor_dir = os.path.join(out_root, g.anchor)
        os.makedirs(anchor_dir, exist_ok=True)
        np.save(os.path.join(anchor_dir, f"{g.label}.npy"), feats.astype(np.float16))
        with open(os.path.join(anchor_dir, f"{g.label}.tsv"), "w") as fh:
            fh.write("id\n")
            for i in ids:
                fh.write(f"{i}\n")

        meta = asdict(g)
        meta.update({"n_seqs": len(seqs), "n_total_in_fasta": len(records),
                     "hidden_dim": int(feats.shape[1]), "seconds": round(dt, 1),
                     "seqs_per_sec": round(rate, 1), "model": loaded_ckpt,
                     "variant": args.variant, "smoke": args.smoke})
        manifest[f"{g.anchor}/{g.label}"] = meta
        json.dump(manifest, open(manifest_path, "w"), indent=2, sort_keys=True)

        print(f"  {g.anchor}/{g.label:24s} n={len(seqs):6d}/{len(records):6d} "
              f"dim={feats.shape[1]} {dt:7.1f}s {rate:8.1f} seq/s")

    print(f"\nwrote {len(pending)} groups -> {out_root}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
