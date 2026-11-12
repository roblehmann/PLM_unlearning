"""Shared test fixtures. Adds the repo root to sys.path (so `import src.*` works
without `pip install -e .`) and builds a tiny toy dataset on disk once per session
by running the same pipeline functions Phase 1 uses."""
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def built_data(tmp_path_factory):
    """Synthesize toy F/F'/R, run the (python-proxy) leakage filter, build splits and
    homologs — mirrors scripts/01_build_data.py at toy scale. Returns a namespace with
    .paths (DataPaths), .raw (paths), .report (leakage report)."""
    from src.data.acquire import synthesize_toy
    from src.data.datamodule import DataPaths, random_split
    from src.data.homologs import generate_homologs
    from src.data.leakage_filter import filter_retain
    from src.utils.io import read_fasta, write_fasta

    d = tmp_path_factory.mktemp("toydata")
    cfg = SimpleNamespace(retain={"n_sequences": 200})
    raw = synthesize_toy(str(d), cfg, seed=0)
    paths = DataPaths(str(d))

    report = filter_retain(
        retain_fasta=raw["retain"],
        family_fastas={"F": raw["F_seqs"], "F_prime": raw["Fp_seqs"]},
        out_fasta=str(paths.retain_filtered),
        max_identity=0.30,
        coverage=0.50,
        tool="python",
    )

    F_tr, F_te = random_split(read_fasta(raw["F_seqs"]), 0.7, 0)
    Fp_tr, Fp_te = random_split(read_fasta(raw["Fp_seqs"]), 0.7, 0)
    write_fasta(F_tr, paths.F_train); write_fasta(F_te, paths.F_test)
    write_fasta(Fp_tr, paths.Fp_train); write_fasta(Fp_te, paths.Fp_test)

    retain_recs = read_fasta(paths.retain_filtered)
    r_tr, r_ho = random_split(retain_recs, 0.8, 0)
    write_fasta(r_tr, paths.retain_train); write_fasta(r_ho, paths.retain_holdout)
    write_fasta(read_fasta(raw["background"]), paths.background)

    generate_homologs(str(paths.F_train), raw["F_hmm"], 3, str(paths.homologs("F")), tool="hmmemit", seed=0)
    generate_homologs(str(paths.Fp_train), raw["Fp_hmm"], 3, str(paths.homologs("F_prime")), tool="hmmemit", seed=0)

    return SimpleNamespace(paths=paths, raw=raw, report=report, root=str(d))
