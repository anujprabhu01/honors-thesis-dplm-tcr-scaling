#!/usr/bin/env python3
"""Parallel Pretrain -> SFT -> Generation Pipeline for DPLM Scaling Experiments.

Each group (model_size, gpu_id) runs pretrain -> SFT -> optional generation
sequentially on its assigned GPU. All groups run in parallel across GPUs.

USAGE
-----
1) Create a groups file (one line per model size):

   groups_batch1.txt        groups_batch2.txt
   -----------------        -----------------
   0.1m,0                   50m,0
   1m,1
   5m,2
   15m,3

2) Run batch 1 (4 models in parallel, 1 GPU each):

   python scripts/scaling/run_scaling_pipeline.py \\
       --groups groups_batch1.txt \\
       --dataset 8m \\
       [--epitopes_file epitopes.txt] \\
       [--num_seqs 100]

3) Run batch 2 after batch 1 completes:

   python scripts/scaling/run_scaling_pipeline.py \\
       --groups groups_batch2.txt \\
       --dataset 8m \\
       [--epitopes_file epitopes.txt] \\
       [--num_seqs 100]

DATASET ARGUMENT
----------------
--dataset controls which experiment configs and log directories are used:
  - "500k"  -> experiment=scaling/tcr_{size}_500k, logs/tcr_{size}_500k/
  - "2m"    -> experiment=scaling/tcr_{size}_2m,   logs/tcr_{size}_2m/
  - "8m"    -> experiment=scaling/tcr_{size}_8m,   logs/tcr_{size}_8m/
  - "32m"   -> experiment=scaling/tcr_{size}_32m,  logs/tcr_{size}_32m/

GENERATION (optional)
---------------------
If --epitopes_file is given, runs generate_conditional.py + fasta_to_csv.py
after SFT for each model. Epitopes file: one epitope per line, blank lines ignored.
Output: results/generated/{dataset}/tcr_{size}_{dataset}/

generate_conditional.py CLI: --checkpoint, --epitopes (space-separated list),
--saveto (output directory), --num_seqs.
fasta_to_csv.py CLI: --input_dir (directory of .fasta files), --output (CSV path).

CHECKPOINT DISCOVERY
--------------------
Looks for best.ckpt in logs/tcr_{size}_{dataset}/checkpoints/ (pretrain)
and logs/tcr_sft_{size}_{dataset}/checkpoints/ (SFT).
"""

import os
import sys
import time
import argparse
import subprocess
import threading
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Groups file parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_groups_file(path: Path) -> list:
    """Parse a groups file. Returns list of (model_size, gpu_id) tuples."""
    groups = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) != 2:
            log(f"WARNING: skipping malformed line: {line!r}")
            continue
        model_size = parts[0].strip()
        try:
            gpu_id = int(parts[1].strip())
        except ValueError:
            log(f"WARNING: skipping malformed line (gpu_id not an integer): {line!r}")
            continue
        groups.append((model_size, gpu_id))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Epitopes file loading
# ─────────────────────────────────────────────────────────────────────────────

def load_epitopes_from_file(path: Path) -> list:
    """Load epitopes from a text file (one per line, blank lines ignored)."""
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(model_size: str, dataset: str, is_sft: bool = False) -> Path:
    """Return best.ckpt for a given model size and dataset.

    Looks in:
      - logs/tcr_{model_size}_{dataset}/checkpoints/best.ckpt  (pretrain)
      - logs/tcr_sft_{model_size}_{dataset}/checkpoints/best.ckpt  (SFT)

    Returns an ABSOLUTE path — required because Hydra changes CWD when
    train.py runs, so relative paths passed as CLI overrides would break.
    """
    prefix = "tcr_sft" if is_sft else "tcr"
    ckpt_dir = Path(f"logs/{prefix}_{model_size}_{dataset}/checkpoints").resolve()
    best = ckpt_dir / "best.ckpt"
    if not best.exists():
        raise FileNotFoundError(f"best.ckpt not found in {ckpt_dir}")
    return best.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Stage runner
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(name: str, cmd: list, gpu_id: int, log_path: Path) -> int:
    """Run cmd on gpu_id, stream output to log_path. Returns exit code."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"START  [{name}]  GPU={gpu_id}  log={log_path}")
    log(f"CMD    {' '.join(str(x) for x in cmd)}")

    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
        proc.wait()

    status = "DONE" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    log(f"{status}  [{name}]")
    try:
        lines = log_path.read_text().splitlines()
        for line in lines[-5:]:
            print(f"    {line}", flush=True)
    except Exception:
        pass
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Group pipeline (pretrain -> SFT -> optional generation)
# ─────────────────────────────────────────────────────────────────────────────

def run_group(
    model_size: str,
    dataset: str,
    gpu_id: int,
    log_dir: Path,
    epitopes: list,
    num_seqs: int,
) -> None:
    """Full pretrain -> SFT -> generation pipeline for one model+dataset on one GPU."""
    tag = f"tcr_{model_size}_{dataset}"

    # ── Pretrain ──────────────────────────────────────────────────────────────
    rc = run_stage(
        name=f"{tag}_pretrain",
        cmd=["python", "train.py",
             f"experiment=scaling/tcr_{model_size}_{dataset}",
             "logger=wandb"],
        gpu_id=gpu_id,
        log_path=log_dir / f"{tag}_pretrain.log",
    )
    if rc != 0:
        log(f"ABORT  [{tag}]  pretrain failed — skipping SFT and generation")
        return

    # ── Find pretrain best checkpoint ─────────────────────────────────────────
    try:
        pretrain_ckpt = find_best_checkpoint(model_size, dataset, is_sft=False)
    except FileNotFoundError as e:
        log(f"ABORT  [{tag}]  {e}")
        return
    log(f"CKPT   [{tag}]  pretrain: {pretrain_ckpt}")

    # ── SFT ───────────────────────────────────────────────────────────────────
    rc = run_stage(
        name=f"{tag}_sft",
        cmd=["python", "train.py",
             f"experiment=scaling/tcr_sft_{model_size}_{dataset}",
             f"model.net.sft_from_checkpoint={pretrain_ckpt}",
             "logger=wandb"],
        gpu_id=gpu_id,
        log_path=log_dir / f"{tag}_sft.log",
    )
    if rc != 0:
        log(f"FAILED [{tag}]  SFT — skipping generation")
        return

    log(f"COMPLETE [{tag}]  pretrain + SFT done")

    # ── Optional: conditional generation ─────────────────────────────────────
    if not epitopes:
        return

    try:
        sft_ckpt = find_best_checkpoint(model_size, dataset, is_sft=True)
    except FileNotFoundError as e:
        log(f"SKIP   [{tag}]  generation — {e}")
        return
    log(f"CKPT   [{tag}]  SFT: {sft_ckpt}")

    gen_dir = Path(f"results/generated/{dataset}/{tag}")
    gen_dir.mkdir(parents=True, exist_ok=True)

    # generate_conditional.py: --checkpoint, --epitopes (space-separated),
    # --saveto (output directory), --num_seqs
    rc = run_stage(
        name=f"{tag}_generate",
        cmd=["python", "scripts/scaling/generate_conditional.py",
             "--checkpoint", str(sft_ckpt),
             "--num_seqs", str(num_seqs),
             "--saveto", str(gen_dir),
             "--epitopes", *epitopes],
        gpu_id=gpu_id,
        log_path=log_dir / f"{tag}_generate.log",
    )
    if rc != 0:
        log(f"FAILED [{tag}]  generation")
        return

    # ── fasta_to_csv ──────────────────────────────────────────────────────────
    # fasta_to_csv.py: --input_dir (directory), --output (CSV path)
    csv_out = Path(f"results/generated/{dataset}/{tag}.csv")
    rc = run_stage(
        name=f"{tag}_fasta_to_csv",
        cmd=["python", "scripts/scaling/fasta_to_csv.py",
             "--input_dir", str(gen_dir),
             "--output", str(csv_out)],
        gpu_id=gpu_id,
        log_path=log_dir / f"{tag}_fasta_to_csv.log",
    )
    if rc != 0:
        log(f"FAILED [{tag}]  fasta_to_csv")
    else:
        log(f"COMPLETE [{tag}]  generation -> {csv_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parallel pretrain->SFT->generation pipeline for DPLM scaling"
    )
    ap.add_argument("--groups", required=True,
                    help="groups file: each line is <model_size>,<gpu_id>")
    ap.add_argument("--dataset", required=True,
                    choices=["500k", "2m", "8m", "32m"],
                    help="Dataset suffix used in config names and log dirs")
    ap.add_argument("--log_dir", default="pipeline_logs",
                    help="Directory for per-stage log files (default: pipeline_logs)")
    ap.add_argument("--epitopes_file", default=None,
                    help="Optional text file with one epitope per line. "
                         "If given, runs generate_conditional.py + fasta_to_csv.py after SFT.")
    ap.add_argument("--num_seqs", type=int, default=100,
                    help="Sequences per epitope for generation (default: 100)")
    args = ap.parse_args()

    groups = parse_groups_file(Path(args.groups))
    if not groups:
        log("ERROR: no valid groups found in groups file")
        sys.exit(1)

    epitopes = []
    if args.epitopes_file:
        epitopes = load_epitopes_from_file(Path(args.epitopes_file))
        log(f"Loaded {len(epitopes)} epitopes from {args.epitopes_file}")

    log(f"Dataset: {args.dataset}")
    log(f"Loaded {len(groups)} group(s):")
    for size, gpu in groups:
        log(f"  tcr_{size}_{args.dataset} -> GPU {gpu}")
    if epitopes:
        log(f"Generation: {len(epitopes)} epitopes, {args.num_seqs} seqs each")
    else:
        log("Generation: disabled (no --epitopes_file given)")
    log("All groups will run in parallel on their assigned GPUs")

    log_dir = Path(args.log_dir)
    threads = [
        threading.Thread(
            target=run_group,
            args=(size, args.dataset, gpu, log_dir, epitopes, args.num_seqs),
            name=f"tcr_{size}_{args.dataset}",
            daemon=False,
        )
        for size, gpu in groups
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log("ALL GROUPS COMPLETED")


if __name__ == "__main__":
    main()
