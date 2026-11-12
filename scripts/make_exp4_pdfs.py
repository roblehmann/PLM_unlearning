#!/usr/bin/env python
"""Publication-ready vector PDFs for the experiment-4 (footprint-vs-generator) figures.

Reuses the committed plotting (aggregate.main_figure / truncation_figure) UNCHANGED, so
the design is identical to the PNGs, but writes vector PDF with embedded TrueType fonts
and a canvas scaled down (via a subplots shim) so the figure sits at LaTeX column width
with legible text. Reads results/summary.json, which stores the full ``arms`` dict.

    python scripts/make_exp4_pdfs.py            # scale 0.62 (author ~6.8in wide)
    EXP4_SCALE=0.7 python scripts/make_exp4_pdfs.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = str(Path(__file__).resolve().parents[1] / "experiments" / "4_footprint_vs_generator")
RES = f"{HERE}/results"
sys.path.insert(0, HERE)

SCALE = float(os.environ.get("EXP4_SCALE", "0.62"))
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

# Scale every figure's canvas without touching the (explicit) font sizes, so the same
# design renders smaller → its text is relatively larger at \textwidth placement.
_orig_subplots = plt.subplots


def _scaled_subplots(*args, **kwargs):
    fs = kwargs.get("figsize")
    if fs:
        kwargs["figsize"] = (fs[0] * SCALE, fs[1] * SCALE)
    return _orig_subplots(*args, **kwargs)


plt.subplots = _scaled_subplots

import aggregate as A  # noqa: E402  (module-level import only; main() is not run)


def main():
    summ = json.load(open(f"{RES}/summary.json"))
    arms = summ["arms"]
    k_report = summ.get("k_report", 32)
    p1 = A.main_figure(arms, Path(f"{RES}/fig_footprint_vs_generator.pdf"), k_report)
    p2 = A.truncation_figure(arms, Path(f"{RES}/fig_truncation_curves.pdf"))
    for p in (p1, p2):
        if p:
            print(f"  wrote {os.path.basename(p)}")
    print(f"canvas scale = {SCALE:.2f}; recommend \\includegraphics[width=\\textwidth]{{...}}")


if __name__ == "__main__":
    main()
