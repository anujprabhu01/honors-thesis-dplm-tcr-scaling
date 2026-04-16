# Modifications from Original DPLM

This document tracks all changes made to the original [bytedance/dplm](https://github.com/bytedance/dplm) repository for TCR scaling law experiments.

---

## Modified Files (from original DPLM)

### `src/byprot/models/utils.py`

**NetConfig dataclass:**
- Made `name: Optional[str]` (was required str pointing to ESM2 pretrained model)
- Added 6 custom architecture fields (all `Optional`, default `None`):
  `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`
- Added `sft_from_checkpoint: Optional[str]` — path to pretrained checkpoint to load before SFT vocab resize

**`get_net()` function:**
- When `name is None`: builds `EsmConfig` from custom architecture fields instead of loading a pretrained config
  - Sets `config._name_or_path = "facebook/esm2_t6_8M_UR50D"` so the tokenizer resolves correctly
  - `vocab_size` defaults to 33, `max_position_embeddings` defaults to 1026
- When `sft_from_checkpoint` is set (SFT runs):
  1. Recreates net with old `vocab_size=33` to match checkpoint shapes
  2. Loads checkpoint weights (strips `"model.net."` prefix from state_dict keys)
  3. Calls `resize_token_embeddings(new_vocab_size)` to expand embeddings
  4. Manually fixes LM head bias: copies old bias values for tokens 0–32, zero-inits token 33

**Why:** Enables training arbitrary model sizes without being constrained to ESM2 checkpoint configs; supports SFT vocabulary expansion for the `<sep>` conditional token.

---

### `src/byprot/models/dplm/dplm.py`

**`compute_loss()` method:**
- After computing `maskable_mask` (non-special positions), applies `condition_mask` from batch:
  ```python
  condition_mask = batch.get("condition_mask", None)
  if condition_mask is not None:
      maskable_mask = maskable_mask & ~condition_mask
  ```
- `condition_mask=True` for `[cls, epitope, sep]` positions → these are excluded from masking
- `condition_mask=False` for `[tcr, eos]` positions → these are masked/predicted normally
- Modified `maskable_mask` propagates into both `q_sample_coupled` and `q_sample` calls

**Backward compatible:** `condition_mask` defaults to `None`; unconditional pretraining is unaffected.

**Why:** During SFT, the epitope side of `<cls> EPITOPE <sep> TCR <eos>` must never be masked or predicted — only the TCR side participates in the diffusion objective.

---

### `src/byprot/datamodules/dataset/uniref_hf.py`

**`load_dataset_from_hf()` function:**
- Added check: if `data_path` is a local directory, uses `load_from_disk(data_path)` then indexes the split
- Otherwise falls through to original `load_dataset(data_path, split=split)` (HuggingFace Hub)

**Why:** TCR datasets are prepared locally with `save_to_disk()` (not uploaded to HuggingFace Hub).

---

### `vendor/openfold/setup.py`

**`get_cuda_bare_metal_version()` function:**
- If `OPENFOLD_CPU_ONLY=1` env var is set, returns `(None, -1, 0)` immediately, skipping CUDA detection

**Why:** Allows OpenFold installation on CPU-only machines and HPC login nodes without CUDA.

---

## New Files

### Data Modules

| File | Description |
|------|-------------|
| `src/byprot/datamodules/tcr_conditional.py` | `TCRConditionalDataModule` — Lightning DataModule for SFT; wraps `TCRConditionalDataset`, uses `ApproxBatchSampler` for token-budget batching |
| `src/byprot/datamodules/dataset/tcr_conditional.py` | `TCRConditionalDataset` (loads HuggingFace dataset with epitope/tcr/length columns); `DPLMCollaterConditional` (builds `<cls> EPITOPE <sep> TCR <eos>` sequences with `condition_mask`); `setup_conditional_dataloader()` |

### Datamodule Configs

The four new Phase 2 pretrain datamodule configs (`tcr_500k.yaml`, `tcr_2m.yaml`, `tcr_8m.yaml`, `tcr_32m.yaml`) all use `_target_: uniref50_hf`, `max_tokens: 4096`, `max_len: 64`, `num_workers: 4`, differing only in `data_dir`. The legacy `tcr.yaml` uses `max_tokens: 6000` (overridden to 4096 in the Phase 1 experiment configs).

| File | `data_dir` | Dataset size |
|------|------------|--------------|
| `configs/datamodule/tcr_500k.yaml` | `${paths.data_dir}/tcr_500k` | 500K sequences (400K train / 100K val) |
| `configs/datamodule/tcr.yaml` | `${paths.data_dir}/tcr_2m` | 2M sequences (1.6M train / 400K val) — Phase 1 legacy alias |
| `configs/datamodule/tcr_2m.yaml` | `${paths.data_dir}/tcr_2m` | 2M sequences (1.6M train / 400K val) |
| `configs/datamodule/tcr_8m.yaml` | `${paths.data_dir}/tcr_8m` | 8M sequences (6.4M train / 1.6M val) |
| `configs/datamodule/tcr_32m.yaml` | `${paths.data_dir}/tcr_32m` | 32M sequences (25.6M train / 6.4M val) |
| `configs/datamodule/tcr_sft.yaml` | `${paths.data_dir}/tcr_sft` | SFT datamodule — `tcr_conditional` target, `data-bin/tcr_sft/`, max_tokens=4096, max_len=64 |

### Experiment Configs — Phase 1 Pretraining (legacy, 2M dataset implied)

All pretrain configs: `random_mask` noise, `linear` weight, AdamW (betas=[0.9, 0.98], weight_decay=0.01), polynomial LR schedule (lr_end=1e-6, warmup_init_lr=1e-7), 1000 warmup steps, 40K max steps, val every 500 steps, `enable_progress_bar: false`. **Note:** Phase 1 actual training runs completed at 20K steps; commit `ca216b2` subsequently updated all pretrain config files (both Phase 1 and Phase 2) to 40K steps.

| File | ~Params | hidden | layers | heads | intermediate | LR |
|------|---------|--------|--------|-------|--------------|-----|
| `configs/experiment/scaling/tcr_smoke_test.yaml` | ~40K | 128 | 2 | 4 | 512 | 1e-4 |
| `configs/experiment/scaling/tcr_0.1m.yaml` | 110K | 48 | 2 | 4 | 192 | 3e-4 |
| `configs/experiment/scaling/tcr_1m.yaml` | 950K | 128 | 4 | 4 | 512 | 1e-4 |
| `configs/experiment/scaling/tcr_5m.yaml` | 5.1M | 256 | 6 | 8 | 1024 | 5e-5 |
| `configs/experiment/scaling/tcr_15m.yaml` | 14.6M | 384 | 8 | 8 | 1536 | 3e-5 |
| `configs/experiment/scaling/tcr_50m.yaml` | ~50M | 640 | 10 | 10 | 2560 | 1e-5 |

LR scaling follows LR ~ N^{-0.5}. Width-dominant: head_dim increases 12→32→32→48→64 across sizes. All use `vocab_size=33`, `max_position_embeddings=1026`, `dropout=0.1`, `pretrain: false`.

### Experiment Configs — Phase 1 SFT (legacy, 2M dataset implied)

All SFT configs: same architecture as pretrain counterpart, `vocab_size=34` (adds `<sep>` token), 500 warmup steps, 20K max steps, val every 500 steps. LR scaled ~1/3 of pretrain LR.

| File | LR | `sft_from_checkpoint` |
|------|----|-----------------------|
| `configs/experiment/scaling/tcr_sft_0.1m.yaml` | 1e-4 | `${paths.root_dir}/logs/tcr_0.1m/checkpoints/best.ckpt` |
| `configs/experiment/scaling/tcr_sft_1m.yaml` | 3e-5 | null (injected at runtime by pipeline) |
| `configs/experiment/scaling/tcr_sft_5m.yaml` | 2e-5 | null |
| `configs/experiment/scaling/tcr_sft_15m.yaml` | 1e-5 | null |
| `configs/experiment/scaling/tcr_sft_50m.yaml` | 3e-6 | null |

### Experiment Configs — Phase 2 Pretraining (5×4 grid, generated by `generate_configs.py`)

All Phase 2 pretrain configs use the same hyperparameters as Phase 1 (same LR per model size, same architecture), but with `max_steps: 40000` (doubled from Phase 1's 20K) and the dataset suffix in both the config name and the `defaults` datamodule. Generated programmatically by `scripts/scaling/generate_configs.py`.

| Size | 500K config | 2M config | 8M config | 32M config |
|------|-------------|-----------|-----------|------------|
| 0.1M | `tcr_0.1m_500k.yaml` | `tcr_0.1m_2m.yaml` | `tcr_0.1m_8m.yaml` | `tcr_0.1m_32m.yaml` |
| 1M   | `tcr_1m_500k.yaml`   | `tcr_1m_2m.yaml`   | `tcr_1m_8m.yaml`   | `tcr_1m_32m.yaml`   |
| 5M   | `tcr_5m_500k.yaml`   | `tcr_5m_2m.yaml`   | `tcr_5m_8m.yaml`   | `tcr_5m_32m.yaml`   |
| 15M  | `tcr_15m_500k.yaml`  | `tcr_15m_2m.yaml`  | `tcr_15m_8m.yaml`  | `tcr_15m_32m.yaml`  |
| 50M  | `tcr_50m_500k.yaml`  | `tcr_50m_2m.yaml`  | `tcr_50m_8m.yaml`  | `tcr_50m_32m.yaml`  |

All 20 files are in `configs/experiment/scaling/`.

### Experiment Configs — Phase 2 SFT (5×4 grid, generated by `generate_configs.py`)

All Phase 2 SFT configs: same architecture and LR as Phase 1 SFT counterparts, `vocab_size=34`, 500 warmup steps, `max_steps: 20000`, `sft_from_checkpoint: null` (injected at runtime). Use the fixed `tcr_sft` datamodule (same SFT data for all runs).

| Size | 500K config | 2M config | 8M config | 32M config |
|------|-------------|-----------|-----------|------------|
| 0.1M | `tcr_sft_0.1m_500k.yaml` | `tcr_sft_0.1m_2m.yaml` | `tcr_sft_0.1m_8m.yaml` | `tcr_sft_0.1m_32m.yaml` |
| 1M   | `tcr_sft_1m_500k.yaml`   | `tcr_sft_1m_2m.yaml`   | `tcr_sft_1m_8m.yaml`   | `tcr_sft_1m_32m.yaml`   |
| 5M   | `tcr_sft_5m_500k.yaml`   | `tcr_sft_5m_2m.yaml`   | `tcr_sft_5m_8m.yaml`   | `tcr_sft_5m_32m.yaml`   |
| 15M  | `tcr_sft_15m_500k.yaml`  | `tcr_sft_15m_2m.yaml`  | `tcr_sft_15m_8m.yaml`  | `tcr_sft_15m_32m.yaml`  |
| 50M  | `tcr_sft_50m_500k.yaml`  | `tcr_sft_50m_2m.yaml`  | `tcr_sft_50m_8m.yaml`  | `tcr_sft_50m_32m.yaml`  |

All 20 files are in `configs/experiment/scaling/`.

### Experiment Configs — LR Sanity Check (50M / 8M)

Two additional configs to determine the optimal LR for the 50M model on the 8M dataset (the 50M model's Phase 1 LR of 1e-5 was designed for the 2M dataset; a higher LR might be better with more data).

| File | LR | Outcome |
|------|----|---------|
| `configs/experiment/scaling/tcr_50m_8m_lr_high.yaml` | 3e-5 | **Winner** — used as the official 50M/8M checkpoint |
| `configs/experiment/scaling/tcr_50m_8m_lr_low.yaml` | 3e-6 | Loser — trained but not used in final results |

Both configs are identical to `tcr_50m_8m.yaml` except for `train.lr`. Both use `max_steps: 40000`.

In WandB exports, `tcr_50m_8m_lr_high` is parsed by `plot_scaling.py` as the `(50m, 8m, False)` entry (the `_lr_high` suffix is stripped). The standard `tcr_50m_8m` run (LR=1e-5, the generated config) was superseded by `tcr_50m_8m_lr_high`.

### Scripts

| File | Description |
|------|-------------|
| `scripts/scaling/prepare_tcr_data.py` | Loads TCR data (pickle or TSV/CSV), samples N sequences with configurable seed, 80/20 train/val split, saves as HuggingFace dataset. CLI: `--source_path`, `--num_samples` (required), `--output_dir` (required), `--seed` (default 9999), `--tcr_column` (defaults to `tcr` for pickle, `junction_aa` for TSV/CSV). For TSV/CSV, filters `productive == "True"` rows and discards non-alpha sequences. |
| `scripts/scaling/generate_configs.py` | One-off script generating the full 5×4 Phase 2 experiment grid: 20 pretrain configs (`tcr_{size}_{dataset}.yaml`) and 20 SFT configs (`tcr_sft_{size}_{dataset}.yaml`). Uses string templates with architecture/LR lookup dicts. LRs are formatted as decimal strings (e.g. `"0.00005"`) to ensure YAML parses them as floats, not strings. |
| `scripts/scaling/prepare_sft_data.py` | Parses `EPITOPE$TCR<EOS>` text files, saves HuggingFace dataset to `data-bin/tcr_sft/` with `{epitope, tcr, length}` fields |
| `scripts/scaling/inspect_tcr_data.py` | Diagnostic: prints shape, length stats, and example sequences from raw TCR pickle |
| `scripts/scaling/generate_conditional.py` | Generates TCRs conditioned on epitopes using SFT checkpoint; tokenizes as `<cls> EPITOPE <sep> <mask>*L <eos>`, fixes epitope via `partial_masks`, runs discrete diffusion decoding, saves FASTA with headers `>EPITOPE=<epi>\|SEQ_<idx>\|L=<len>` |
| `scripts/scaling/fasta_to_csv.py` | Converts generation FASTA → CSV; parses `EPITOPE=` tags from headers; filters non-alpha sequences to remove `<sep>`/`<unk>` leakage; outputs `TCRs,Epitopes` columns |
| `scripts/scaling/run_scaling_pipeline.py` | Orchestrates parallel pretrain→SFT→generation across GPUs. CLI: `--groups` (groups file), `--dataset` (one of `500k/2m/8m/32m`), `--log_dir`, `--epitopes_file` (optional), `--num_seqs` (default 100). Each (model_size, gpu_id) pair from the groups file runs as a Python thread with `CUDA_VISIBLE_DEVICES` set. Pipeline: pretrain → find `best.ckpt` → SFT (injects absolute checkpoint path as Hydra override) → optional generation → fasta_to_csv. Checkpoint discovery looks in `logs/tcr_{size}_{dataset}/checkpoints/best.ckpt`. |
| `scripts/scaling/plot_scaling.py` | Generates all 5 thesis scaling law figures from WandB CSV exports + ashour evaluation CSVs. See details below. |
| `scripts/scaling/plot_attention_maps.py` | Extracts and visualizes last-layer attention (TCR rows × epitope cols) across all 5 SFT model sizes; uses forward hooks on `EsmSelfAttention` to manually compute Q·K^T softmax (needed because `F.scaled_dot_product_attention` discards weights) |
| `scripts/scaling/compute_ll_scores.py` | Original TCR-BERT log-likelihood script (legacy, not used for final results) |
| `scripts/scaling/compute_ll_scores_mod.py` | Modified TCR-BERT PLL script used on ashour: adds `tcrbert_mlm_ll_masking` column (true pseudo-log-likelihood — each position masked individually). Modified copy of `compute_ll_scores.py` (both files coexist); loads models locally instead of from HuggingFace Hub. |

### `scripts/scaling/plot_scaling.py` — Detailed Description

**Purpose:** Reads all WandB summary CSVs and evaluation CSVs; fits power-law curves; produces 5 figures saved to `results/figures/`.

**Constants:**
- `SIZES = ["0.1m", "1m", "5m", "15m", "50m"]`
- `DATASETS = ["500k", "2m", "8m", "32m"]`
- `SIZE_PARAMS`: fallback parameter counts (millions) when WandB export lacks `model/params/total`
- `DATASET_SEQS`: dataset sizes in millions (`500k→0.5`, `2m→2.0`, `8m→8.0`, `32m→32.0`)
- `EVAL_METRICS`: 4 metrics — `bap_cnn`, `bap_lstm`, `tcr_match`, `tcrbert_mlm_ll_masking`

**Key functions:**

`parse_run_name(name)` — Maps a WandB run name to `(size_str, dataset_str, is_sft)` or `None`. Strips `_lr_high`/`_lr_low` suffix via `re.sub`. Handles Phase 1 names (`tcr_50m` → `("50m","2m",False)`) and Phase 2 names (`tcr_sft_5m_8m` → `("5m","8m",True)`).

`load_wandb_summary()` — Searches both `ROOT/*.csv` and `ROOT/"wandb exports"/*.csv`; sorts by filename (chronological); processes in order so later files overwrite earlier entries for the same `(size, dataset, is_sft)` key (last-write-wins). Prefers `"val/nll_loss (Min)"` column over `"val/nll_loss"` via `VAL_COL_CANDIDATES` tuple. Skips files without a `"Name"` column (step-by-step exports). Guards against NaN with `if np.isnan(val_loss): continue`.

`load_eval_scores()` — Reads `results/scored/tcr_{size}_{dataset}.csv`; computes column means for each eval metric per model+dataset combination.

`fit_power_law(x, y)` — scipy `curve_fit` on `L = a·x^{-α} + c`; bounds `a∈[0,∞)`, `α∈[0.01,2.0]`, `c∈(-∞,∞)`; returns `(a, alpha, c)` or `None` on failure.

**Output figures:**

| File | Description |
|------|-------------|
| `results/figures/scaling_law_2m.png` | L(N) at 2M dataset — pretrain and SFT side by side with power-law fits and α annotations |
| `results/figures/scaling_law_nd.png` | L(N) curves at all 4 data sizes on shared axes — pretrain and SFT panels, colored by dataset size |
| `results/figures/ld_curves.png` | L(D) curves at all 5 model sizes — pretrain and SFT panels. NOTE: fits are unreliable because each dataset has its own 20% validation split (different val sets), so cross-dataset val loss is not directly comparable (unlike Kaplan 2020 / Chinchilla which hold the eval set constant). |
| `results/figures/eval_vs_n.png` | 4-panel figure: each panel shows one eval metric (BAP-CNN, BAP-LSTM, TCRMatch, TCR-BERT PLL) vs model size, one curve per dataset size |
| `results/figures/eval_vs_d.png` | 4-panel figure: each panel shows one eval metric vs dataset size (in millions of sequences), one curve per model size |

### Other

| File | Description |
|------|-------------|
| `groups.txt` | GPU assignment for `run_scaling_pipeline.py`; format: `model_size,gpu_id` per line |

---

## Data

### Pretraining — Phase 1 (2M)

- **Source**: `/mnt/disk11/user/xiaoyih1/data/tcr_data_all/data/tcr_repertoires_healthy_samples/tcr_repertoire_seqs.pkl`
- **Sampling**: 2M sequences, seed=9999
- **Split**: 80% train (1.6M), 20% val (400K)
- **Output**: `data-bin/tcr_2m/` (HuggingFace dataset, `{seq, length}` columns)

### Pretraining — Phase 2 (500K, 8M, 32M)

- **Source**: `/mnt/disk12/.../trb_combined_backup_20260203.tsv` (AIRR-format TSV, `junction_aa` column)
- **Filter**: `productive == "True"`, then `seq.isalpha()`
- **Sampling**: seed=9999 for all three sizes
- **Splits** (80/20):

| Dataset | Train | Val | Approx. epochs at 40K steps |
|---------|-------|-----|------------------------------|
| 500K    | 400K  | 100K | ~24 epochs |
| 8M      | 6.4M  | 1.6M | ~2 epochs |
| 32M     | 25.6M | 6.4M | ~0.5 epochs |

- **Outputs**: `data-bin/tcr_500k/`, `data-bin/tcr_8m/`, `data-bin/tcr_32m/`
- **Phase 2 2M**: reuses `data-bin/tcr_2m/` prepared in Phase 1 from the pickle — not re-prepared

### SFT

- **Source**: `/mnt/disk11/user/xiaoyih1/data/compact_format/merged_seed42_epi_test_pair_split/` (train.txt, val.txt, test.txt)
- **Format**: `EPITOPE$TCR<EOS>` one pair per line
- **Output**: `data-bin/tcr_sft/` (HuggingFace dataset, `{epitope, tcr, length}` columns)
- **Used by all SFT runs** (the SFT data is fixed across all model sizes and pretrain dataset sizes)

---

## Results Structure

```
results/
  figures/
    scaling_law_2m.png      # L(N) at 2M — pretrain + SFT power-law fits
    scaling_law_nd.png      # L(N) at all 4 data sizes — pretrain + SFT panels
    ld_curves.png           # L(D) at all 5 model sizes (confounded — see notes)
    eval_vs_n.png           # 4 eval metrics vs model size
    eval_vs_d.png           # 4 eval metrics vs dataset size
    attention_maps.png      # Last-layer attention heatmaps across model sizes
    scaling_law.png         # Legacy Phase 1 figure (superseded)
    pretrain_loss_curves.png  # Legacy
    sft_loss_curves.png       # Legacy
  scored/
    tcr_{size}_{dataset}.csv  # Eval scores from ashour for each of 20 SFT models
    # 20 files: sizes {0.1m,1m,5m,15m,50m} × datasets {500k,2m,8m,32m}
    # Columns: tcr, epi, bap_cnn, bap_lstm, tcr_match,
    #          tcrbert_ll_all, tcrbert_ll_masking,
    #          tcrbert_mlm_ll_all, tcrbert_mlm_ll_masking
  sft_{size}_test_epis.csv    # Phase 1 generation CSVs (legacy, superseded)
```

---

## Non-Obvious Implementation Details

**Vocabulary resize for SFT:** `resize_token_embeddings()` randomly initializes new token embeddings. We manually copy the pretrained LM head bias for tokens 0–32 and zero-init token 33 (`<sep>`) to preserve pretrained behavior for existing tokens.

**Condition mask logic:** `condition_mask=True` means "do not mask this position." The bitwise `maskable_mask & ~condition_mask` double-negation is easy to misread. True in condition_mask = excluded from diffusion.

**Tokenizer `<sep>` registration:** `generate_conditional.py`, `plot_attention_maps.py`, and `DPLMCollaterConditional` all call `tokenizer.add_tokens([SEP_TOKEN])`. This is idempotent — calling it multiple times is safe.

**Hydra CWD:** `train.py` uses Hydra which changes the working directory at startup. `generate_conditional.py` sets `os.environ["PROJECT_ROOT"] = os.getcwd()` before any imports to capture the true project root for config resolution.

**Checkpoint paths in pipeline:** SFT configs have `sft_from_checkpoint: null`; `run_scaling_pipeline.py` injects the absolute path as a Hydra CLI override after pretrain completes. Relative paths would resolve against Hydra's CWD and fail.

**Attention extraction:** `EsmSelfAttention` in `dplm_modeling_esm.py` uses `F.scaled_dot_product_attention` which does not return attention weights. `plot_attention_maps.py` registers a forward hook that manually computes `softmax(Q·K^T / sqrt(d_k))` from the inputs to the self-attention module.

**FASTA column naming:** `fasta_to_csv.py` outputs `TCRs,Epitopes` (not `tcr,epi`). This is required for compatibility with `ensemble_bap.py` and `compute_ll_scores_mod.py` on ashour which hardcode these column names.

**Min val/nll_loss vs last-step:** All training runs save `best.ckpt` (the checkpoint with minimum validation loss). Conditional generation is performed with `best.ckpt`. WandB default exports report the last-step value; the April 15 WandB export uses "Min" aggregation to report the true minimum, which is consistent with which checkpoint is actually used for evaluation.

**Primary evaluation metric:** `tcrbert_mlm_ll_masking` is the correct pseudo-log-likelihood. It masks each position individually and sums the log probabilities — the proper MLM PLL formulation. The other TCR-BERT columns (`tcrbert_ll_all`, `tcrbert_ll_masking`, `tcrbert_mlm_ll_all`) are non-proper because they allow bidirectional context leakage, overestimating sequence likelihood.

**L(D) confounding:** The four pretraining datasets (500K, 2M, 8M, 32M) each have their own 20% validation split, so the validation set grows with dataset size. Cross-dataset val loss comparisons are not valid (unlike Kaplan 2020 and Chinchilla/Hoffmann 2022 which hold the evaluation set constant). As a result, power-law fits to L(D) fail — β hits the lower bound (0.01) — and `ld_curves.png` shows non-monotonic patterns. This is a known limitation of the experimental design.

**Phase 2 pretrain max_steps:** Phase 2 configs use `max_steps: 40000` (double Phase 1's 20K) to give larger datasets more training tokens. Phase 2 SFT configs retain `max_steps: 20000` (same as Phase 1 SFT, since SFT data is fixed).

**LR format in generated configs:** `generate_configs.py` formats LRs as `f"{lr:.10f}".rstrip("0").rstrip(".")` (e.g. `3e-5` → `"0.00003"`). Scientific notation like `3e-05` is parsed by YAML as a string, not a float, which would break Hydra's type coercion.

**tcr_50m_8m_lr_high as canonical 50M/8M:** The generated `tcr_50m_8m.yaml` config (LR=1e-5) was superseded by the LR sanity-check winner `tcr_50m_8m_lr_high` (LR=3e-5). In `plot_scaling.py`, `parse_run_name` strips the `_lr_high` suffix so this run contributes to the `(50m, 8m, False)` data point. Both the standard and `_lr_high` runs may appear in older WandB exports; last-write-wins in the export processing ensures the correct one is used.
