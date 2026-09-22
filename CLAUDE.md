# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a master's thesis codebase studying the **reintegration phenomenon** in federated multimodal emotion recognition. The research question: when a model trained exclusively on full multimodal (audio+text) data encounters intermittent modality absence, does prediction quality dip at the boundary where the modality returns?

The codebase is a local extension package (`reintegration`) that builds on the upstream `fed_multimodal` package without modifying it. It runs on the **IEMOCAP** and **MELD** datasets.

## Key Concepts

- **Scene**: a conversation (sequence of T utterances), the unit of forward pass
- **Client**: each speaker in the dataset is a federated learning client
- **Markov mask**: deterministic per-utterance availability mask (1=present, 0=absent) generated from scene ID seed with configurable transition probabilities
- **Reintegration event**: where `mask[t-1]==0` and `mask[t]==1` (modality returns after absence)
- **Stable-only training**: the model is trained with all-ones mask (modality always present). The reintegration dip is measured at eval time by running each scene twice — once stable, once masked — and comparing predictions at reintegration boundaries

## Architecture

### Model (`reintegration/model/mm_models.py`)

Two nested modules:
- **`SERClassifier`**: utterance-level encoder — Conv1D+GRU for audio (MFCC), GRU for text (MobileBERT), fusion via `FuseBaseSelfAttention` (multi-head cross-modal attention concatenated), then a classifier head
- **`SceneGRUWrapper`**: wraps SERClassifier with a cross-utterance GRU. The scene GRU's hidden state carries absence history, which is the mechanism through which reintegration effects manifest

### Dataloading (`reintegration/dataloader/`)

- **`SceneDataset`** (`scene_dataloader.py`): yields one scene per batch (batch_size=1). Each batch is `(scene_x_a, scene_x_b, scene_len_a, scene_len_b, scene_labels, scene_mask)` — lists of T per-utterance tensors
- **`DataloadManager`** (`dataload_manager.py`): loads per-client audio/text feature pickles, builds scene dataloaders via `set_scene_dataloader()`. Also handles legacy unimodal dataloading
- Markov masks are deterministic: `generate_markov_mask(scene_len, p_stay_absent, p_stay_present, seed=hash(scene_id))`

### Training (`reintegration/trainers/`)

- **`Server`** (`server_trainer.py`): manages FL rounds (client sampling, weight aggregation, logging, checkpointing). Contains `run_reintegration_eval()` — the core evaluation method that runs stable vs masked passes and computes per-timestep recovery curves, UAR deltas, log-prob gaps, KL divergence, and FuseBase attention entropy
- **`ClientFedAvg`** (`fed_avg_trainer.py`): stable-only local training (all-ones mask). No masked pass during training

### Evaluation (`reintegration/evaluation.py`)

- **`EvalMetric`**: accumulates per-batch predictions, computes epoch-level loss, accuracy, top-5 accuracy, UAR (macro recall), and macro F1

### Feature Extraction (`reintegration/features/`)

- **`extract_audio_feature.py`**: MFCC extraction (Kaldi fbank, 80-dim, global z-score normalized per client)
- **`extract_text_feature.py`**: MobileBERT extraction (512-dim, no normalization)
- **`data_partition.py`**: loads MELD/IEMOCAP CSVs, builds scene-nested partition.json via Dirichlet split across clients

### Constants (`reintegration/constants/constants.py`)

- `feature_len_dict`: maps feature names to dimensions (mfcc=80, mobilebert=512)
- `num_class_dict`: maps dataset names to class counts (iemocap=6, meld=7)

## Running Commands

### Cluster training (SLURM)

The project runs on a remote SLURM cluster. Job scripts set the working directory to `$SCRATCH/reintegration-aware-mm-fusion` (the full project, not just this extensions dir):

```bash
# Standard training (submit via `sbatch submit_job.sh` on cluster)
python -m reintegration.train \
    --en_text_only \
    --mask_modality text \
    --dataset iemocap \
    --data_dir reintegration/output \
    --modality multimodal \
    --audio_feat mfcc \
    --text_feat mobilebert \
    --fed_alg fed_avg \
    --num_epochs 60 \
    --local_epochs 1 \
    --sample_rate 1.0 \
    --batch_size 16 \
    --hid_size 128 \
    --learning_rate 0.01 \
    --en_att \
    --att_name fuse_base \
    --availability_process markov

# Holdout client experiment
python -m reintegration.train \
    --holdout_clients 8 9 \
    --en_audio_only \
    --mask_modality audio \
    ...

# Eval-only mode (no training, load checkpoint)
python -m reintegration.train \
    --eval_only \
    --ckpt_path "path/to/model.pt" \
    ...
```

Key CLI flags:
- `--holdout_clients 8 9` — exclude clients from FL training, evaluate post-training
- `--run_name "mask_text_ses5"` — append to output path to avoid overwriting parallel runs
- `--reint_reset_scene_hidden` — ablation: reset scene GRU hidden state each utterance
- `--reint_collect_fuse_attention` — collect FuseBase attention entropy at reintegration timesteps
- `--reint_save_timestep_detail` — save per-timestep recovery rows to JSON

### Bootstrap post-processing (local)

```bash
# Run bootstrap via shell script (local)
./run_bootstrap.sh

# Or manually:
python -m reintegration.scripts.bootstrap_reintegration_json \
    --json reintegration/output/partition/.../reintegration_detailed_fold1.json \
    --split test --offset 0 --seed 0

# Summarize bootstrap results across folds:
python -m reintegration.scripts.summarize_bootstrap_results \
    --glob 'reintegration/output/.../fold1_summary/boot_fold1_off*.json' \
    --out-csv summary.csv \
    --out-plot summary.png \
    --print-table
```

### Unit tests

```bash
python -m pytest reintegration/tests/test_set_scene_dataloader.py -v
```

Tests validate the `SceneDataset` batch contract: 6-tuple output, tensor shapes, Markov mask dtype, missing feature fallback, and short-audio handling.

## Data Layout

```
reintegration/output/
  partition/{dataset}/partition.json      # scene-nested client assignment
  feature/audio/mfcc/{dataset}/{client}.pkl   # {filename: ndarray(T, 80)}
  feature/text/mobilebert/{dataset}/{client}.pkl  # {filename: ndarray(T, 512)}
  log/{fed_alg}/{dataset}/{feature}/{att}/{settings}/fold{N}/  # checkpoints + tensorboard
  result/{fed_alg}/{dataset}/{feature}/{att}/{settings}/      # result.json
```

The `partition.json` structure: `{client_id: [scene], ...}` where `scene = [utterance]` and `utterance = [Filename, Path, Label, Utterance_text, None]`.

## Important Notes

- **Exclude `masters_venv/`** — this is a local venv, not project code
- **`reintegration/output/`** is mostly generated (features, logs, partitions) and gitignored
- The project's parent working directory on the cluster is `$SCRATCH/reintegration-aware-mm-fusion`; this repo (`my_extensions`) provides the `reintegration` package
- Configuration is loaded from `reintegration/system.cfg` (simple key=value), with defaults overridden by CLI args
- Scene batch_size is always 1 (scenes have variable T, cannot be stacked)
