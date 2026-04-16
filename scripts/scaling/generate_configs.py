#!/usr/bin/env python
"""One-off script to generate all experiment configs for the 5x4 scaling grid.

Generates configs for all 5 model sizes × 4 data sizes (2M, 500K, 8M, 32M):
  - 20 pretrain configs (tcr_{size}_{dataset}.yaml)
  - 20 SFT configs     (tcr_sft_{size}_{dataset}.yaml)

Run from repo root:
    python scripts/scaling/generate_configs.py

Output: configs/experiment/scaling/tcr_{size}_{dataset}.yaml (and tcr_sft_...)
"""

from pathlib import Path

CONFIGS_DIR = Path("configs/experiment/scaling")

# Architecture per model size (identical to Phase 1)
ARCH = {
    "0.1m": dict(hidden_size=48,  num_hidden_layers=2,  num_attention_heads=4,  intermediate_size=192),
    "1m":   dict(hidden_size=128, num_hidden_layers=4,  num_attention_heads=4,  intermediate_size=512),
    "5m":   dict(hidden_size=256, num_hidden_layers=6,  num_attention_heads=8,  intermediate_size=1024),
    "15m":  dict(hidden_size=384, num_hidden_layers=8,  num_attention_heads=8,  intermediate_size=1536),
    "50m":  dict(hidden_size=640, num_hidden_layers=10, num_attention_heads=10, intermediate_size=2560),
}

# Learning rates per model size (N^{-0.5} scaling, identical to Phase 1)
PRETRAIN_LR = {"0.1m": 3e-4, "1m": 1e-4, "5m": 5e-5, "15m": 3e-5, "50m": 1e-5}
SFT_LR      = {"0.1m": 1e-4, "1m": 3e-5, "5m": 2e-5, "15m": 1e-5, "50m": 3e-6}

# All 5 model sizes × 4 data sizes
DATASETS = {
    "0.1m": ["2m", "500k", "8m", "32m"],
    "1m":   ["2m", "500k", "8m", "32m"],
    "5m":   ["2m", "500k", "8m", "32m"],
    "15m":  ["2m", "500k", "8m", "32m"],
    "50m":  ["2m", "500k", "8m", "32m"],
}

PRETRAIN_TEMPLATE = """\
# @package _global_

# TCR scaling experiment: ~{size_label} parameters, {dataset_label} dataset
# Usage: python train.py experiment=scaling/tcr_{size}_{dataset} logger=wandb

defaults:
  - /datamodule: tcr_{dataset}
  - /callbacks: lm
  - /trainer: ddp_bf16

project: "TCR_Scaling"
name: "tcr_{size}_{dataset}"

datamodule:
  max_tokens: 4096
  max_len: 64
  num_workers: 4

model:
  _target_: dplm
  num_diffusion_timesteps: 500
  gradient_ckpt: false
  rdm_couple: false
  lora:
    enable: false
  net:
    arch_type: esm
    name: null
    hidden_size: {hidden_size}
    num_hidden_layers: {num_hidden_layers}
    num_attention_heads: {num_attention_heads}
    intermediate_size: {intermediate_size}
    vocab_size: 33
    max_position_embeddings: 1026
    dropout: 0.1
    pretrain: false

task:
  _target_: lm/dplm
  learning:
    noise: random_mask
    watch_t1_t2_loss: false
    cal_constant_loss: false
    weight: linear
  criterion:
    _target_: byprot.modules.cross_entropy.RDMCrossEntropyLoss
    label_smoothing: 0.0
    ignore_index: 1
  optimizer:
    type: adamw
    _partial_: true
    lr: ${{train.lr}}
    betas:
      - 0.9
      - 0.98
    weight_decay: 0.01
  lr_scheduler:
    type: polynomial
    warmup_steps: 1000
    total_steps: ${{trainer.max_steps}}
    lr: ${{train.lr}}
    lr_end: 1e-6
    warmup_init_lr: 1e-07
    power: 1

train:
  seed: 42
  lr: {lr}
  monitor: "val/loss"
  mode: "min"
  patience: 1000

trainer:
  min_epochs: 1
  max_epochs: 10000
  gradient_clip_val: 0.0
  num_sanity_val_steps: 0
  reload_dataloaders_every_n_epochs: 1
  use_distributed_sampler: false
  max_steps: 40000
  accumulate_grad_batches: 1
  check_val_every_n_epoch: null
  val_check_interval: 500
  enable_progress_bar: false
  num_nodes: 1
"""

SFT_TEMPLATE = """\
# @package _global_

# TCR SFT experiment: ~{size_label} parameters, pretrained on {dataset_label} dataset
# Fine-tunes pretrained DPLM on epitope-TCR pairs
# NOTE: sft_from_checkpoint is injected by run_scaling_pipeline.py at runtime

defaults:
  - /datamodule: tcr_sft
  - /callbacks: lm
  - /trainer: ddp_bf16

project: "TCR_Scaling"
name: "tcr_sft_{size}_{dataset}"

datamodule:
  max_tokens: 4096
  max_len: 64
  num_workers: 4

model:
  _target_: dplm
  num_diffusion_timesteps: 500
  gradient_ckpt: false
  rdm_couple: false
  lora:
    enable: false
  net:
    arch_type: esm
    name: null
    hidden_size: {hidden_size}
    num_hidden_layers: {num_hidden_layers}
    num_attention_heads: {num_attention_heads}
    intermediate_size: {intermediate_size}
    vocab_size: 34
    max_position_embeddings: 1026
    dropout: 0.1
    pretrain: false
    sft_from_checkpoint: null  # injected by run_scaling_pipeline.py at runtime

task:
  _target_: lm/dplm
  learning:
    noise: random_mask
    watch_t1_t2_loss: false
    cal_constant_loss: false
    weight: linear
  criterion:
    _target_: byprot.modules.cross_entropy.RDMCrossEntropyLoss
    label_smoothing: 0.0
    ignore_index: 1
  optimizer:
    type: adamw
    _partial_: true
    lr: ${{train.lr}}
    betas:
      - 0.9
      - 0.98
    weight_decay: 0.01
  lr_scheduler:
    type: polynomial
    warmup_steps: 500
    total_steps: ${{trainer.max_steps}}
    lr: ${{train.lr}}
    lr_end: 1e-6
    warmup_init_lr: 1e-07
    power: 1

train:
  seed: 42
  lr: {sft_lr}
  monitor: "val/loss"
  mode: "min"
  patience: 1000

trainer:
  min_epochs: 1
  max_epochs: 10000
  gradient_clip_val: 0.0
  num_sanity_val_steps: 0
  reload_dataloaders_every_n_epochs: 1
  use_distributed_sampler: false
  max_steps: 20000
  accumulate_grad_batches: 1
  check_val_every_n_epoch: null
  val_check_interval: 500
  enable_progress_bar: false
  num_nodes: 1
"""

SIZE_LABELS = {
    "0.1m": "0.1M", "1m": "1M", "5m": "5M", "15m": "15M", "50m": "50M"
}
DATASET_LABELS = {
    "2m": "2M", "500k": "500K", "8m": "8M", "32m": "32M"
}


def main():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    created = []

    for size, datasets in DATASETS.items():
        arch = ARCH[size]
        pretrain_lr = PRETRAIN_LR[size]
        sft_lr = SFT_LR[size]

        for dataset in datasets:
            size_label = SIZE_LABELS[size]
            dataset_label = DATASET_LABELS[dataset]

            # Format LRs as decimal notation so YAML parses them as floats
            # e.g. 5e-5 -> "0.00005" rather than "5e-05" (which YAML treats as string)
            lr_str = f"{pretrain_lr:.10f}".rstrip("0").rstrip(".")
            sft_lr_str = f"{sft_lr:.10f}".rstrip("0").rstrip(".")

            # Pretrain config
            pretrain_path = CONFIGS_DIR / f"tcr_{size}_{dataset}.yaml"
            pretrain_path.write_text(PRETRAIN_TEMPLATE.format(
                size=size, dataset=dataset,
                size_label=size_label, dataset_label=dataset_label,
                lr=lr_str, **arch
            ))
            created.append(str(pretrain_path))

            # SFT config
            sft_path = CONFIGS_DIR / f"tcr_sft_{size}_{dataset}.yaml"
            sft_path.write_text(SFT_TEMPLATE.format(
                size=size, dataset=dataset,
                size_label=size_label, dataset_label=dataset_label,
                sft_lr=sft_lr_str, **arch
            ))
            created.append(str(sft_path))

    print(f"Created {len(created)} config files:")
    for f in sorted(created):
        print(f"  {f}")


if __name__ == "__main__":
    main()
