# Constraints + Transfer Experiment Checklist

This file is a master tracker for:

- standard mixed pretraining on `data/mixed` with zero-mode and PDE constraints
- BC-conditioned mixed pretraining on `data/bc/mixed` with BC enforcement variants
- downstream fine-tuning on Poisson, AdvDiff, and Helmholtz
- optional custom joint BC + zero-mode + PDE configs if those are added later

Assumption for this study:

- Constraints are applied only during mixed pretraining.
- Downstream fine-tuning reuses the pretrained checkpoint but does not introduce new constraint settings.

Repo ground truth for this checklist:

- Standard mixed constrained presets: `config/operators_mixed.yaml`
- Standard mixed sweep docs: `docs/transfer-learning/FOUNDATION_CONSTRAINTS_EXPERIMENT_RUNBOOK.md`
- BC-conditioned mixed presets: `config/operators_mixed_bc.yaml`
- BC walkthrough: `docs/transfer-learning/BC_MIXED_PRETRAIN_SOFT_HARD_WALKTHROUGH.md`
- Mixed transfer helper: `scripts/maintenance/update_mixed_checkpoint_path.sh`
- Downstream configs: `config/operators_poisson.yaml`, `config/operators_ad.yaml`, `config/operators_helmholtz.yaml`

## 1. Scope and gating

- [ ] Decide whether to run the recommended shortlist or the full exhaustive matrix.
- [ ] Record one canonical location for sweep IDs, SLURM job IDs, run names, and checkpoint paths.
- [ ] Decide whether joint BC + zero-mode + PDE configs will be added to `config/operators_mixed_bc.yaml`.

Recommended shortlist to send to transfer:

- [ ] `mixed-scale-all` baseline
- [ ] best PDE-only winner
- [ ] best zero-only winner
- [ ] best zero-mode + PDE winner
- [ ] best BC winner
- [ ] best joint BC + zero-mode + PDE winner, if custom configs are added

Exhaustive transfer scale if every predefined fixed variant is pushed downstream:

- `13` predefined pretraining variants x `24` downstream trainings each = `312` fine-tuning runs for the full `16, 64, 256, 1k, 4k, 8k, 16k, 32k` curve.
- `13` predefined pretraining variants x `12` downstream trainings each = `156` fine-tuning runs for the low-data-only `16, 64, 256, 1k` study.

## 2. Shared setup

- [ ] Mixed standard dataset exists at `data/mixed/_train_mixed_32k.h5` with scales file.
- [ ] BC datasets exist for Poisson, AdvDiff, and Helmholtz if the BC track will be run.
- [ ] Mixed BC dataset exists at `data/bc/mixed/_train_mixed_32k_bc.h5` with scales file.
- [ ] Standard constraints smoke path has been sanity-checked.
- [ ] BC constraints smoke path has been sanity-checked.
- [ ] Decide whether downstream variance should come from shuffle only or from both shuffle + random subset selection.

Suggested smoke commands:

```bash
bash scripts/experiments/submit_constraints_sanity_sweep.sh
MODES="off soft hard hard+soft" bash scripts/workflows/run_local_smoke_train_eval_bc_constraints.sh
```

Seeded downstream runs are now available as an opt-in path and do not change legacy pretraining behavior unless you pass the new flags.

Recommended paper-style downstream flags:

- `--seed <N>`: override the run seed without editing YAML
- `--train_shuffle`: enable seeded minibatch shuffling for downstream training
- `--random_train_subset`: when `subsample > 1`, choose a seeded random subset instead of the legacy deterministic stride subset
- `--subset_seed <N>`: optional separate seed for subset selection; defaults to `--seed`

Example direct launch for one seeded downstream run:

```bash
python scripts/entrypoints/train.py \
  --yaml_config config/operators_poisson.yaml \
  --config poisson-k1_2.5-finetune-mixed-256 \
  --run_num transfer-finetune-mixed-256-seed3 \
  --root_dir experiments \
  --seed 3 \
  --train_shuffle \
  --random_train_subset
```

Example multi-seed evaluation after runs finish:

```bash
python scripts/entrypoints/eval_transfer_learning.py \
  --yaml_config config/operators_poisson.yaml \
  --experiment_type poisson \
  --configs poisson-k1_2.5-finetune-mixed-256 \
  --experiment_dir experiments \
  --output_dir results/transfer_learning_seeded \
  --aggregate_runs \
  --run_pattern '*-seed*'
```

## 3. Standard mixed pretraining tracker

### 3.1 Baseline mixed pretraining

- [ ] Run unconstrained mixed baseline: `mixed-scale-all`
- [ ] Record baseline SLURM job ID
- [ ] Record baseline checkpoint path

Default launch:

```bash
sbatch scripts/slurm/pretrain/submit_pretrain_mixed.sh
```

### 3.2 Zero-mode-only track

- [ ] Optional helper comparison submitted: `bash scripts/experiments/submit_constraints_zero_mode_only_compare.sh`
- [ ] Run hard zero-only baseline: `mixed-scale-all-constraints-zero-hard-only`
- [ ] Run soft zero-only sweep: `config/sweep_constraints_pretrain_zero_soft_only.yaml`
- [ ] Select zero-only soft candidate
- [ ] Promote zero-only soft winner into a final config or explicit override
- [ ] Run final soft zero-only confirmation
- [ ] Choose zero-only winner for transfer

### 3.3 PDE-only track

- [ ] Run PDE-only penalty sweep: `config/sweep_constraints_pretrain_penalty_pde_only.yaml`
- [ ] Run PDE-only AL sweep: `config/sweep_constraints_pretrain_al_pde_only.yaml`
- [ ] Rank PDE-only sweep outputs
- [ ] Promote PDE-only winner settings into a final config or explicit override
- [ ] Run final PDE-only confirmation run(s)
- [ ] Choose PDE-only winner for transfer

### 3.4 Combined zero-mode + PDE track

- [ ] Run `penalty + hard zero-mode` sweep: `config/sweep_constraints_pretrain_penalty_hard.yaml`
- [ ] Run `AL + hard zero-mode` sweep: `config/sweep_constraints_pretrain_al_hard.yaml`
- [ ] Select Stage-B winning PDE method
- [ ] Run matching soft zero-mode sweep for the winning PDE method
- [ ] Rank combined sweep outputs
- [ ] Promote combined winner settings into a final config or explicit override

If running the full fixed preset set, track all four combined presets:

- [ ] `mixed-scale-all-constraints-penalty-hard`
- [ ] `mixed-scale-all-constraints-al-hard`
- [ ] `mixed-scale-all-constraints-penalty-soft`
- [ ] `mixed-scale-all-constraints-al-soft`

If running the strict staged protocol only, track just the winner-side final confirmations:

- [ ] Winner hard preset final run complete
- [ ] Winner soft preset final run complete
- [ ] Combined zero-mode + PDE winner chosen for transfer

### 3.5 Standard mixed sweep bookkeeping

- [ ] All relevant `SWEEP_ID`s recorded
- [ ] Ranking JSONs saved under `results/constraints/`
- [ ] Final standard mixed pretraining checkpoints logged with config name, run name, and job ID

## 4. BC-conditioned mixed pretraining tracker

### 4.1 BC dataset preparation

- [ ] Generate BC datasets: `bash scripts/workflows/run_gen_data_bc.sh`
- [ ] Build mixed BC dataset: `bash scripts/workflows/run_build_mixed_bc.sh`
- [ ] Run BC dataset sanity checks

### 4.2 Fixed BC mode comparison

- [ ] Run `mixed-bc-scale-all-off`
- [ ] Run `mixed-bc-scale-all-soft`
- [ ] Run `mixed-bc-scale-all-hard`
- [ ] Run `mixed-bc-scale-all-hard-soft`
- [ ] Compare `val_err`, `val_err_interior`, `val_bc_violation_raw`, `val_bc_violation_final`
- [ ] Choose fixed-mode BC leader

Convenience entry points:

```bash
bash scripts/experiments/submit_bc_constraints_mode_off.sh
bash scripts/experiments/submit_bc_constraints_mode_soft.sh
bash scripts/experiments/submit_bc_constraints_mode_hard.sh
bash scripts/experiments/submit_bc_constraints_mode_hard_soft.sh
```

### 4.3 BC soft sweep

- [ ] Run BC soft sweep: `config/sweep_constraints_pretrain_bc_soft.yaml`
- [ ] Rank BC soft sweep outputs
- [ ] Choose BC soft winner
- [ ] Promote BC soft winner into a final config or explicit override

### 4.4 Final BC checkpoint selection

- [ ] Run final BC fixed-mode winner confirmation
- [ ] Run final BC soft winner confirmation, if soft remains competitive
- [ ] Choose BC winner for transfer
- [ ] Record BC pretraining config name, run name, job ID, and checkpoint path

## 5. Joint BC + zero-mode + PDE combinations

Current repo status:

- Standard mixed configs already support zero-mode + PDE combinations in `config/operators_mixed.yaml`.
- BC-conditioned configs in `config/operators_mixed_bc.yaml` currently define BC enforcement variants only.
- Joint BC + zero-mode + PDE configs are not pre-defined yet.

If you plan to add those custom combinations, track them here:

- [ ] Define custom joint config name 1: `________________`
- [ ] Define custom joint config name 2: `________________`
- [ ] Define any required custom sweep YAMLs
- [ ] Run custom joint sweeps
- [ ] Rank custom joint sweeps
- [ ] Run final custom joint confirmation run(s)
- [ ] Choose custom joint winner for transfer

## 6. Pretraining checkpoint handoff into transfer configs

Do this once for every selected mixed-pretraining checkpoint that should be evaluated downstream.

- [ ] Record `PRETRAIN_CONFIG_NAME`
- [ ] Record `PRETRAIN_RUN_PREFIX`
- [ ] Record `JOB_ID`
- [ ] Record `RUN_INDEX` if not `0`
- [ ] Record final checkpoint path
- [ ] Update mixed-transfer weights in downstream configs
- [ ] Verify the updated weights in `config/operators_poisson.yaml`
- [ ] Verify the updated weights in `config/operators_ad.yaml`
- [ ] Verify the updated weights in `config/operators_helmholtz.yaml`

Default helper:

```bash
bash scripts/maintenance/update_mixed_checkpoint_path.sh <JOBID>
```

Override example for constrained or BC-conditioned pretraining:

```bash
PRETRAIN_CONFIG_NAME=<CONFIG_NAME> \
PRETRAIN_RUN_PREFIX=<RUN_PREFIX> \
bash scripts/maintenance/update_mixed_checkpoint_path.sh <JOBID>
```

## 7. Zero-shot preview tracker

Use this stage after checkpoint propagation and before launching the full downstream fine-tuning bundles.

Purpose:

- sanity-check that the transferred checkpoint is usable on each downstream PDE family
- get a rough idea of cross-domain behavior before paying for the full transfer matrix
- catch obviously weak checkpoints before launching all fine-tuning jobs

Protocol note:

- If you want to keep the strict foundation-model protocol from the constraints runbook, treat zero-shot as a diagnostic/reporting stage only.
- Do not use downstream zero-shot to tune pretraining hyperparameters if you want to avoid downstream leakage.

These mixed zero-shot configs already exist:

- `poisson-k1_2.5-zeroshot-mixed`
- `ad-adr0p2_0p4-zeroshot-mixed`
- `helm-o1_5-zeroshot-mixed`

They will pick up the current mixed-pretraining checkpoint path from the downstream YAMLs after `update_mixed_checkpoint_path.sh` is run.

Convenience bundle script for multiple standard mixed checkpoints:

```bash
bash scripts/eval/run_mixed_zeroshot_bundle.sh \
  mixed-scale-all:pretrain-mixed:<JOBID> \
  mixed-scale-all-constraints-al-hard:pretrain-mixed-al-hard:<JOBID>
```

Suggested zero-shot preview commands:

```bash
python scripts/entrypoints/eval.py --yaml_config config/operators_poisson.yaml --config poisson-k1_2.5-zeroshot-mixed --run_num zeroshot-preview-poisson-<tag> --root_dir experiments
python scripts/entrypoints/eval.py --yaml_config config/operators_ad.yaml --config ad-adr0p2_0p4-zeroshot-mixed --run_num zeroshot-preview-advdiff-<tag> --root_dir experiments
python scripts/entrypoints/eval.py --yaml_config config/operators_helmholtz.yaml --config helm-o1_5-zeroshot-mixed --run_num zeroshot-preview-helmholtz-<tag> --root_dir experiments
```

What to record:

- `test_err`
- rough ranking across Poisson, AdvDiff, and Helmholtz
- whether the checkpoint looks promising enough to justify the full fine-tuning bundle

### 7.1 Zero-shot preview: baseline mixed

Pretrain config: `mixed-scale-all`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson zero-shot done
- [ ] AdvDiff zero-shot done
- [ ] Helmholtz zero-shot done
- [ ] Zero-shot notes recorded

### 7.2 Zero-shot preview: best zero-only winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson zero-shot done
- [ ] AdvDiff zero-shot done
- [ ] Helmholtz zero-shot done
- [ ] Zero-shot notes recorded

### 7.3 Zero-shot preview: best PDE-only winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson zero-shot done
- [ ] AdvDiff zero-shot done
- [ ] Helmholtz zero-shot done
- [ ] Zero-shot notes recorded

### 7.4 Zero-shot preview: best zero-mode + PDE winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson zero-shot done
- [ ] AdvDiff zero-shot done
- [ ] Helmholtz zero-shot done
- [ ] Zero-shot notes recorded

### 7.5 Zero-shot preview: best BC winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson zero-shot done
- [ ] AdvDiff zero-shot done
- [ ] Helmholtz zero-shot done
- [ ] Zero-shot notes recorded

### 7.6 Zero-shot preview: best joint BC + zero-mode + PDE winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson zero-shot done
- [ ] AdvDiff zero-shot done
- [ ] Helmholtz zero-shot done
- [ ] Zero-shot notes recorded

## 8. Downstream fine-tuning bundle tracker

Each selected pretraining checkpoint expands into downstream transfer bundles across three datasets.

Per-checkpoint full-curve bundle:

- Poisson: `16, 64, 256, 1k, 4k, 8k, 16k, 32k`
- AdvDiff: `16, 64, 256, 1k, 4k, 8k, 16k, 32k`
- Helmholtz: `16, 64, 256, 1k, 4k, 8k, 16k, 32k`

Bundle scripts:

- Poisson low-data plus 4k: `scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh`
- Poisson small split: `scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_small.sh`
- Poisson medium split: `scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_medium.sh`
- Poisson large split: `scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_large.sh`
- AdvDiff small split: `scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_small.sh`
- AdvDiff medium split: `scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_medium.sh`
- AdvDiff large split: `scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_large.sh`
- Helmholtz small split: `scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_small.sh`
- Helmholtz medium split: `scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_medium.sh`
- Helmholtz large split: `scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_large.sh`

Important note:

- The Poisson mixed medium script currently only launches `8k`; the `4k` line is commented out there.
- If you need `4k` for Poisson, use the mixed array script or edit that medium split script before relying on it.

### 8.1 Transfer bundle: baseline mixed

Pretrain config: `mixed-scale-all`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson small bundle done
- [ ] Poisson medium bundle done
- [ ] Poisson large bundle done
- [ ] AdvDiff small bundle done
- [ ] AdvDiff medium bundle done
- [ ] AdvDiff large bundle done
- [ ] Helmholtz small bundle done
- [ ] Helmholtz medium bundle done
- [ ] Helmholtz large bundle done
- [ ] Poisson eval job done
- [ ] AdvDiff eval job done
- [ ] Helmholtz eval job done

### 8.2 Transfer bundle: best zero-only winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson small bundle done
- [ ] Poisson medium bundle done
- [ ] Poisson large bundle done
- [ ] AdvDiff small bundle done
- [ ] AdvDiff medium bundle done
- [ ] AdvDiff large bundle done
- [ ] Helmholtz small bundle done
- [ ] Helmholtz medium bundle done
- [ ] Helmholtz large bundle done
- [ ] Poisson eval job done
- [ ] AdvDiff eval job done
- [ ] Helmholtz eval job done

### 8.3 Transfer bundle: best PDE-only winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson small bundle done
- [ ] Poisson medium bundle done
- [ ] Poisson large bundle done
- [ ] AdvDiff small bundle done
- [ ] AdvDiff medium bundle done
- [ ] AdvDiff large bundle done
- [ ] Helmholtz small bundle done
- [ ] Helmholtz medium bundle done
- [ ] Helmholtz large bundle done
- [ ] Poisson eval job done
- [ ] AdvDiff eval job done
- [ ] Helmholtz eval job done

### 8.4 Transfer bundle: best zero-mode + PDE winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson small bundle done
- [ ] Poisson medium bundle done
- [ ] Poisson large bundle done
- [ ] AdvDiff small bundle done
- [ ] AdvDiff medium bundle done
- [ ] AdvDiff large bundle done
- [ ] Helmholtz small bundle done
- [ ] Helmholtz medium bundle done
- [ ] Helmholtz large bundle done
- [ ] Poisson eval job done
- [ ] AdvDiff eval job done
- [ ] Helmholtz eval job done

### 8.5 Transfer bundle: best BC winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson small bundle done
- [ ] Poisson medium bundle done
- [ ] Poisson large bundle done
- [ ] AdvDiff small bundle done
- [ ] AdvDiff medium bundle done
- [ ] AdvDiff large bundle done
- [ ] Helmholtz small bundle done
- [ ] Helmholtz medium bundle done
- [ ] Helmholtz large bundle done
- [ ] Poisson eval job done
- [ ] AdvDiff eval job done
- [ ] Helmholtz eval job done

### 8.6 Transfer bundle: best joint BC + zero-mode + PDE winner

Pretrain config: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson small bundle done
- [ ] Poisson medium bundle done
- [ ] Poisson large bundle done
- [ ] AdvDiff small bundle done
- [ ] AdvDiff medium bundle done
- [ ] AdvDiff large bundle done
- [ ] Helmholtz small bundle done
- [ ] Helmholtz medium bundle done
- [ ] Helmholtz large bundle done
- [ ] Poisson eval job done
- [ ] AdvDiff eval job done
- [ ] Helmholtz eval job done

### 8.7 Extra transfer bundles for exhaustive comparisons

Duplicate this block if you decide to transfer every predefined fixed preset instead of only the shortlist.

Pretrain config: `________________`
Run prefix: `________________`
Job ID: `________________`

- [ ] Checkpoint propagated with `update_mixed_checkpoint_path.sh`
- [ ] Poisson small bundle done
- [ ] Poisson medium bundle done
- [ ] Poisson large bundle done
- [ ] AdvDiff small bundle done
- [ ] AdvDiff medium bundle done
- [ ] AdvDiff large bundle done
- [ ] Helmholtz small bundle done
- [ ] Helmholtz medium bundle done
- [ ] Helmholtz large bundle done
- [ ] Poisson eval job done
- [ ] AdvDiff eval job done
- [ ] Helmholtz eval job done

## 9. Final evaluation and reporting

- [ ] Poisson transfer results archived
- [ ] AdvDiff transfer results archived
- [ ] Helmholtz transfer results archived
- [ ] Pretraining-side selection metrics archived
- [ ] Transfer curves generated for every selected checkpoint
- [ ] Low-data mean over `16, 64, 256, 1k` computed
- [ ] Full-curve mean over `16, 64, 256, 1k, 4k, 8k, 16k, 32k` computed
- [ ] Constraint diagnostics reviewed where relevant
- [ ] Final winner chosen
- [ ] Final write-up notes captured

## 10. Quick reference: predefined pretraining variants

Standard mixed:

- `mixed-scale-all`
- `mixed-scale-all-constraints-zero-hard-only`
- `mixed-scale-all-constraints-zero-soft-only`
- `mixed-scale-all-constraints-penalty-pde-only`
- `mixed-scale-all-constraints-al-pde-only`
- `mixed-scale-all-constraints-penalty-hard`
- `mixed-scale-all-constraints-al-hard`
- `mixed-scale-all-constraints-penalty-soft`
- `mixed-scale-all-constraints-al-soft`

BC-conditioned mixed:

- `mixed-bc-scale-all-off`
- `mixed-bc-scale-all-soft`
- `mixed-bc-scale-all-hard`
- `mixed-bc-scale-all-hard-soft`
