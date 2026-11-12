# taskblock-poc

Code for measuring how durable a weight-level safeguard on a protein language model is
against fine-tuning. A safeguard is applied to ESM-2 so that the model loses its ability
to model one protein family (the *forget family*, F). An attacker then fine-tunes the
model on F with increasing amounts of data, and the harness reports how much more data
the attacker needs than against a control model that received the same extra compute but
no safeguard.

This repository accompanies a scientific paper and documents the experiments reported
there. It contains code, configurations and job scripts. It does not contain data, model
checkpoints or result files; all of these are produced by running the pipeline.

All experiments use benign surrogate protein families (common enzyme families from Pfam
and uncharacterised families from MGnify). No hazardous sequences are used or needed.

## What the harness measures

Three models are compared, all starting from the same ESM-2 checkpoint:

| Model | How it is made |
|---|---|
| `A` | the unmodified ESM-2 base model |
| `B` | base model plus the safeguard phase |
| `A'` | base model plus ordinary masked-language-model training, matched to `B` in compute |

Two safeguards are implemented:

- **Task blocking**, a first-order MLAC/TAR-style method
  ([src/train/taskblock.py](src/train/taskblock.py)).
- **CIR**, Collapse of Irrelevant Representations ([src/train/cir.py](src/train/cir.py)).

The main quantities are:

- **Acquisition:** pseudo-perplexity on held-out sequences of F. Lower means the model
  has acquired the family.
- **Threshold τ:** `τ = PPL_naive − tau_rho · (PPL_naive − PPL_floor)`. `PPL_naive` is
  the base model's pseudo-perplexity on F, and `PPL_floor` is the value reached after
  fine-tuning the base model on all of the F training set.
- **Multiplier M:** the smallest number of attacker training sequences at which a model
  crosses τ, as a ratio between models. The headline ratio is `B / A'`. `A' / A` is a
  validity check and should be close to 1. If a model never crosses τ within the largest
  budget, M is reported as a lower bound.
- **Utility retention:** relative difference in pseudo-perplexity between `B` and `A'`
  on held-out retain sequences.
- **Durability under harder attacks:** M when the attacker's data is augmented with
  synthetic homologs, and M for a sibling family F'.

Thresholds are set in [configs/metrics/thresholds.yaml](configs/metrics/thresholds.yaml).

## Repository layout

```
configs/        Hydra configs: model, data, blocking, attack, sweep, metrics
src/
  data/         sequence download, leakage filter, homolog generation, data module
  models/       ESM-2 loading, optional LoRA
  train/        masked-LM training, attacker, task blocking, CIR
  eval/         pseudo-perplexity, membership probe, utility
  sweep/        attacker sweep grid and sharding
  analysis/     multiplier with right-censoring, weight-space analysis, plots
  utils/        experiment tracking, seeding, FLOP accounting, I/O
scripts/        one driver per pipeline phase, plus figure scripts
experiments/    self-contained additional experiments (0 to 7)
slurm/          sbatch scripts for the pipeline phases, and env_setup.sh
tests/          unit tests
```

## Installation

```bash
conda env create -f environment.yml
conda activate taskblock-poc
pip install -e .
```

The environment installs `mmseqs2`, `hmmer` and `cd-hit` from bioconda. These are used
for the leakage filter and for homolog generation.

Install a PyTorch build that matches the CUDA version of your GPU driver. If the build is
too new for the driver, `torch.cuda.is_available()` returns `False` and the harness falls
back to CPU without an error.

ESM-2 weights are downloaded from the Hugging Face Hub on first use
(`facebook/esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`,
`esm2_t33_650M_UR50D`).

## Quick check without downloads

This builds a small synthetic dataset and runs one training step on the 8M model. It
needs no network access beyond the model download and runs on CPU.

```bash
TASKBLOCK_ROOT=$PWD python scripts/01_build_data.py data.toy=true tracking.backend=none
python scripts/00_smoke.py model=esm2_8m tracking.backend=none device=cpu
```

## Main pipeline

Each phase is one script in `scripts/` with a matching job script in `slurm/`. Run them
in order. The baseline phase is a gate: it checks that the base model has enough room to
improve on F for the multiplier to be measurable, and the later phases should only be run
if it passes.

| Phase | Script | Produces |
|---|---|---|
| Smoke test | `scripts/00_smoke.py` | one training step; confirms the configs resolve |
| Data | `scripts/01_build_data.py` | F, F' and retain sets, splits, homologs, leakage report |
| Baseline and gate | `scripts/02_baseline_attack.py` | `PPL_naive`, `PPL_floor`, τ, pass or fail |
| Safeguard | `scripts/03_train_blocked.py` or `scripts/03b_train_cir.py` | checkpoint `B` |
| Control | `scripts/04_train_control.py` | checkpoint `A'`, compute-matched to `B` |
| Attacker sweep | `scripts/05_full_sweep.py` | long-format results table |
| Analysis | `scripts/06_analyze.py` | multipliers, plots, summary |

```bash
python scripts/01_build_data.py \
    data.clan.pfam_id=CLxxxx data.forget_family.pfam_id=PFxxxxx data.adjacent_family.pfam_id=PFyyyyy
python scripts/02_baseline_attack.py model=esm2_35m
python scripts/03_train_blocked.py   model=esm2_35m
python scripts/04_train_control.py   model=esm2_35m
python scripts/05_full_sweep.py      model=esm2_35m
python scripts/06_analyze.py         model=esm2_35m
```

Further scripts analyse a finished run:

| Script | Purpose |
|---|---|
| `06b_cir_frontier.py` | efficacy against disruption across CIR operating points |
| `06c_selection_sensitivity.py` | sensitivity of M to the checkpoint selection rule |
| `06d_per_seed_variation.py` | per-seed variation of M |
| `08_weightspace.py` | weight displacement and linear mode connectivity between base and `B` |
| `08b` to `08e` | CIR subspace analyses: projected displacement, attacker repair, F and retain overlap |
| `09_siddiqui_reconcile.py` | M against weight displacement and connectivity barrier |
| `10_sibling_ladder_sweep.py`, `10b_sibling_ladder_analyze.py` | relearning F from related families at graded similarity |
| `11_decon_static_rescore.py` | re-scoring on test sets with train homologs removed |
| `12_embed_corpora.py`, `13_umap_plot.py`, `14_umap_compare.py` | embedding-space views of F, F' and retain sets |
| `07_manuscript.py`, `make_*_pdfs.py`, `plot_*.py` | tables and figures |

### Configuration

Configuration uses [Hydra](https://hydra.cc). Defaults are in
[configs/config.yaml](configs/config.yaml), and any value can be overridden on the
command line, for example `model=esm2_8m` or `attack=full_ft`.

All data, checkpoints and results are written below `paths.root`, which defaults to the
repository root or to `$TASKBLOCK_ROOT`. To keep several runs side by side, give each its
own root:

```bash
python scripts/02_baseline_attack.py paths.root=runs/my_run
```

Experiment tracking defaults to Weights & Biases. Use `tracking.backend=none` to log to
local JSONL files instead.

## Additional experiments

Each directory in `experiments/` is self-contained and has its own job scripts. Where a
`run_all.sh` or `run_ladder.sh` exists, it submits the whole experiment as a chain of
dependent SLURM jobs.

| Directory | Question |
|---|---|
| `0_F_candidate_PPL` | How well does base ESM-2 already model each candidate forget family? |
| `1_naive_PPL_Floor` | How much can fine-tuning improve on each candidate, and is that enough headroom? |
| `2_FR_overlap` | How much do the activation subspaces of a candidate F and the retain set overlap? |
| `3_scale_ladder` | Do the gate, the weight-space diagnostics and M hold across the 8M, 35M, 150M and 650M models? |
| `4_footprint_vs_generator` | Why does the same safeguard produce a strong static effect on one family and a weak one on another? |
| `5_readout_vs_representation` | Does the safeguard act in the output head or in the trunk? Tested by splicing parts of `B` and the base model. |
| `6_mgnify_reps_vs_retain` | Are MGnify family representatives novel to ESM-2, in embedding space and in pseudo-perplexity? |
| `7_pfam_reps_vs_retain` | How do Pfam family representatives relate to the retain set, and which families are suitable as F? |

## Data

`scripts/01_build_data.py` downloads Pfam family sequences through the InterPro API and
a UniRef50 subsample through the UniProt API. The streamed UniRef50 subsample is the
first *n* records, not a uniform sample; for a uniform sample, shuffle a local UniRef50
release.

The runs on MGnify protein families read prepared FASTA files from their run root.
Building these from the public MGnify release is not automated by
`scripts/01_build_data.py`.

## Running on a SLURM cluster

The job scripts are written for SLURM and are submitted from the repository root:

```bash
sbatch slurm/02_baseline_attack.sbatch model=esm2_35m
```

They will need adjusting to your site:

- The GPU request is `--gres=gpu:a100:1` in most scripts.
- [slurm/env_setup.sh](slurm/env_setup.sh) loads `mmseqs2`, `hmmer` and `cdhit` as
  environment modules and activates the conda environment. Change the module names, or
  remove them if you use the bioconda tools.
- Set `TASKBLOCK_CONDA_SH` to your `conda.sh` and `TASKBLOCK_ENV` to the environment name
  if they differ from the defaults.

The attacker sweep runs as a job array, one shard per task. Each shard writes its own
CSV, and `scripts/06_analyze.py` combines them.

## Outputs

```
data/         manifest.json, leakage_report.json, splits/, homologs/, filtered/
checkpoints/  B/, A_prime/
results/      thresholds.json, B_compute.json, Aprime_compute.json, sweep/*.csv,
              multipliers.csv, summary.json, plots/
```

## Tests

```bash
pytest -q            # logic tests; no GPU, network or external tools needed
pytest -q -m slow    # adds tests that build ESM-2 and run short training loops
```

## Limitations

- The task-blocking safeguard uses a first-order approximation. A weak multiplier under
  this approximation is a lower bound for a full second-order implementation.
- The forget families are benign and chosen to be well separated from the retain data.
  This is the easy case for a safeguard, so the measured multipliers are optimistic
  relative to a target that is entangled with benign function.

## References

- Henderson et al., *Self-Destructing Models: Increasing the Costs of Harmful Dual Uses
  of Foundation Models*, AIES 2023.
- Tamirisa et al., *Tamper-Resistant Safeguards for Open-Weight LLMs*, arXiv:2408.00761.
- Sondej and Yang, *Collapse of Irrelevant Representations (CIR) Ensures Robust and
  Non-Disruptive LLM Unlearning*, arXiv:2509.11816, NeurIPS 2025.
- Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a
  language model*, Science 2023 (ESM-2).

## Citation

Citation details will be added when the paper is published.

## License

Released under the MIT License. See [LICENSE](LICENSE).
