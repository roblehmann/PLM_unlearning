"""Phase 1 — data pipeline.

Acquire F, F', R (+ background), run the leakage filter on R against F and F', build
the splits, generate homologs, and persist a manifest + leakage report.

    # offline toy data (no network, no big downloads) — used to exercise the pipeline:
    python scripts/01_build_data.py data.toy=true tracking.backend=none

    # real data (set the Pfam ids first; needs network + mmseqs2/hmmer):
    python scripts/01_build_data.py data.forget_family.pfam_id=PFxxxxx \
        data.adjacent_family.pfam_id=PFyyyyy data.clan.pfam_id=CLzzzz

Done when: F, F', R and all splits exist with counts in the manifest; the leakage
report shows the identity filter was applied and logged; no F/F' homolog above
threshold remains in R.
"""
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 1 — BUILD DATA")
    from src.data.datamodule import DataPaths, random_split
    from src.data.homologs import generate_homologs
    from src.data.leakage_filter import filter_retain
    from src.utils.io import count_fasta, ensure_dir, read_fasta, save_json, write_fasta

    d = cfg.data
    paths = DataPaths(cfg.paths.data)
    ensure_dir(cfg.paths.data)

    # ---- acquire raw F / F' / R / background -------------------------------- #
    if bool(d.toy):
        from src.data.acquire import synthesize_toy

        print("toy mode: synthesizing offline data")
        raw = synthesize_toy(cfg.paths.data, d, seed=cfg.seed)
        F_seqs, Fp_seqs = raw["F_seqs"], raw["Fp_seqs"]
        retain_raw, background_raw = raw["retain"], raw["background"]
        F_hmm, Fp_hmm = raw["F_hmm"], raw["Fp_hmm"]
    else:
        from src.data.acquire import (
            download_family_hmm,
            download_family_sequences,
            download_uniref50_subsample,
        )

        for fam in (d.forget_family.pfam_id, d.adjacent_family.pfam_id):
            if str(fam).upper() == "TBD":
                raise SystemExit("Set data.forget_family.pfam_id and data.adjacent_family.pfam_id "
                                 "(Phase 1 selection criteria) before running real-data mode.")
        rawdir = ensure_dir(Path(cfg.paths.data) / "raw")
        F_seqs = str(rawdir / "F" / "family.fasta")
        Fp_seqs = str(rawdir / "F_prime" / "family.fasta")
        F_hmm = str(paths.hmm("F"))
        Fp_hmm = str(paths.hmm("F_prime"))
        retain_raw = str(rawdir / "retain" / "uniref50_subsample.fasta")
        background_raw = str(rawdir / "background" / "background.fasta")
        f_max = d.forget_family.get("download_max", None)
        fp_max = d.adjacent_family.get("download_max", None)
        print(f"downloading F={d.forget_family.pfam_id} (max {f_max}), "
              f"F'={d.adjacent_family.pfam_id} (max {fp_max}) ...")
        download_family_sequences(d.forget_family.pfam_id, F_seqs, max_n=int(f_max) if f_max else None)
        download_family_sequences(d.adjacent_family.pfam_id, Fp_seqs, max_n=int(fp_max) if fp_max else None)
        download_family_hmm(d.forget_family.pfam_id, F_hmm)
        download_family_hmm(d.adjacent_family.pfam_id, Fp_hmm)
        download_uniref50_subsample(int(d.retain.n_sequences), retain_raw, seed=cfg.seed)
        download_uniref50_subsample(int(d.background.n_sequences), background_raw, seed=cfg.seed + 1)

    # ---- leakage filter: R vs {F, F'} --------------------------------------- #
    print("running leakage filter on R against F and F' ...")
    report = filter_retain(
        retain_fasta=retain_raw,
        family_fastas={"F": F_seqs, "F_prime": Fp_seqs},
        out_fasta=str(paths.retain_filtered),
        max_identity=float(d.leakage_filter.max_identity_to_F),
        coverage=float(d.leakage_filter.coverage),
        tool=str(d.leakage_filter.tool),
        workdir=str(Path(cfg.paths.data) / "tmp_leakage"),
    )
    save_json(report, paths.leakage_report)
    print(f"leakage: in={report['n_in']} removed={report['n_removed']} out={report['n_out']} "
          f"(tool={report['tool']}, max_id<{report['max_identity_to_F']})")

    # ---- splits ------------------------------------------------------------- #
    print("building splits ...")
    ensure_dir(Path(paths.F_train).parent)
    F_tr, F_te = random_split(read_fasta(F_seqs), float(d.splits.forget_train_frac), int(d.splits.seed))
    Fp_tr, Fp_te = random_split(read_fasta(Fp_seqs), float(d.splits.forget_train_frac), int(d.splits.seed))
    write_fasta(F_tr, paths.F_train); write_fasta(F_te, paths.F_test)
    write_fasta(Fp_tr, paths.Fp_train); write_fasta(Fp_te, paths.Fp_test)

    retain_recs = read_fasta(paths.retain_filtered)
    holdout_n = min(max(8, len(retain_recs) // 10), 5000)
    r_tr, r_ho = random_split(retain_recs, 1.0 - holdout_n / max(1, len(retain_recs)), int(d.splits.seed))
    write_fasta(r_tr, paths.retain_train); write_fasta(r_ho, paths.retain_holdout)
    shutil.copyfile(background_raw, paths.background)

    # ---- homologs (durability stressor) ------------------------------------ #
    print("generating homologs from train seeds ...")
    n_per = int(d.homolog_generator.n_per_seed_seq)
    generate_homologs(str(paths.F_train), F_hmm, n_per, str(paths.homologs("F")),
                      tool=str(d.homolog_generator.tool), seed=cfg.seed)
    generate_homologs(str(paths.Fp_train), Fp_hmm, n_per, str(paths.homologs("F_prime")),
                      tool=str(d.homolog_generator.tool), seed=cfg.seed)

    # ---- manifest ----------------------------------------------------------- #
    manifest = {
        "config": OmegaConf.to_container(d, resolve=True),
        "counts": {
            "F_total": count_fasta(F_seqs), "F_train": len(F_tr), "F_test": len(F_te),
            "F_prime_total": count_fasta(Fp_seqs), "F_prime_train": len(Fp_tr), "F_prime_test": len(Fp_te),
            "retain_raw": count_fasta(retain_raw), "retain_filtered": report["n_out"],
            "retain_train": len(r_tr), "retain_holdout": len(r_ho),
            "background": count_fasta(paths.background),
            "F_homologs": count_fasta(paths.homologs("F")),
            "F_prime_homologs": count_fasta(paths.homologs("F_prime")),
        },
        "leakage_report": report,
    }
    save_json(manifest, paths.manifest)

    tracker = _bootstrap.init_tracker_from_cfg(cfg, run_name="build_data")
    tracker.log_summary({"counts": manifest["counts"], "leakage_removed": report["n_removed"]})
    tracker.finish()

    _bootstrap.banner("DATA BUILD DONE")
    print(OmegaConf.to_yaml(OmegaConf.create(manifest["counts"])))
    # acceptance check
    ok = all(manifest["counts"][k] > 0 for k in ("F_train", "F_test", "F_prime_train", "retain_train"))
    print("ACCEPTANCE:", "PASS" if ok else "FAIL", "— splits populated; leakage filter applied & logged.")


if __name__ == "__main__":
    main()
