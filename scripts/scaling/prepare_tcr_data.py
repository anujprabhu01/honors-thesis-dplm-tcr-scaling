#!/usr/bin/env python
"""Prepare TCR data for DPLM pretraining.

Supports pickle (legacy) and TSV/CSV sources. For TSV sources, uses the
AIRR standard `junction_aa` column and filters for productive sequences.

Usage:
    # Original 2M dataset from pickle (wolf):
    python scripts/scaling/prepare_tcr_data.py \
        --num_samples 2000000 --output_dir data-bin/tcr_2m

    # 500K from TSV:
    python scripts/scaling/prepare_tcr_data.py \
        --source_path /mnt/disk12/.../trb_combined_backup_20260203.tsv \
        --num_samples 500000 --output_dir data-bin/tcr_500k

    # 8M from TSV:
    python scripts/scaling/prepare_tcr_data.py \
        --source_path /mnt/disk12/.../trb_combined_backup_20260203.tsv \
        --num_samples 8000000 --output_dir data-bin/tcr_8m

    # 32M from TSV:
    python scripts/scaling/prepare_tcr_data.py \
        --source_path /mnt/disk12/.../trb_combined_backup_20260203.tsv \
        --num_samples 32000000 --output_dir data-bin/tcr_32m
"""

import argparse
import pickle
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

DEFAULT_SOURCE = (
    "/mnt/disk11/user/xiaoyih1/data/tcr_data_all/data/"
    "tcr_repertoires_healthy_samples/tcr_repertoire_seqs.pkl"
)
TRAIN_RATIO = 0.80
VALID_RATIO = 0.20


def parse_args():
    p = argparse.ArgumentParser(description="Prepare TCR HuggingFace dataset")
    p.add_argument("--source_path", default=DEFAULT_SOURCE,
                   help="Path to source file (.pkl/.pickle or .tsv/.csv)")
    p.add_argument("--tcr_column", default=None,
                   help="Column with TCR sequences. Defaults to 'tcr' for pickle, "
                        "'junction_aa' for TSV/CSV.")
    p.add_argument("--num_samples", type=int, required=True,
                   help="Number of sequences to sample")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for the HuggingFace dataset")
    p.add_argument("--seed", type=int, default=9999,
                   help="Random seed for reproducibility (default: 9999)")
    return p.parse_args()


def load_sequences(source: Path, tcr_column: Optional[str]) -> list:
    """Load all valid (productive, alpha-only) TCR sequences from source file."""
    suffix = source.suffix.lower()

    if suffix in (".pkl", ".pickle"):
        col = tcr_column or "tcr"
        print(f"Loading pickle, column='{col}'")
        with open(source, "rb") as f:
            df = pickle.load(f)
        seqs = df[col].dropna().tolist()

    elif suffix in (".tsv", ".csv"):
        col = tcr_column or "junction_aa"
        sep = "\t" if suffix == ".tsv" else ","
        print(f"Loading TSV/CSV, column='{col}', filtering productive==True")
        df = pd.read_csv(source, sep=sep, usecols=[col, "productive"], dtype=str)
        df = df[df["productive"] == "True"]
        seqs = df[col].dropna().tolist()

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .pkl, .tsv, or .csv")

    before = len(seqs)
    seqs = [s for s in seqs if isinstance(s, str) and s.isalpha()]
    dropped = before - len(seqs)
    if dropped:
        print(f"Filtered {dropped:,} non-alpha sequences ({dropped / before * 100:.1f}%)")
    return seqs


def main():
    args = parse_args()
    source = Path(args.source_path)
    output = Path(args.output_dir)

    print(f"Loading source data: {source}")
    all_sequences = load_sequences(source, args.tcr_column)

    print(f"Total valid sequences available: {len(all_sequences):,}")
    if args.num_samples > len(all_sequences):
        raise ValueError(
            f"Requested {args.num_samples:,} samples but only "
            f"{len(all_sequences):,} valid sequences available."
        )
    print(f"Sampling {args.num_samples:,} sequences with seed={args.seed}")

    random.seed(args.seed)
    sequences = random.sample(all_sequences, args.num_samples)

    random.shuffle(sequences)

    n_train = int(len(sequences) * TRAIN_RATIO)
    train_seqs = sequences[:n_train]
    valid_seqs = sequences[n_train:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_seqs):,} ({len(train_seqs)/len(sequences)*100:.1f}%)")
    print(f"  Valid: {len(valid_seqs):,} ({len(valid_seqs)/len(sequences)*100:.1f}%)")

    def make_ds(seqs):
        return Dataset.from_dict({"seq": seqs, "length": [len(s) for s in seqs]})

    dataset_dict = DatasetDict({
        "train": make_ds(train_seqs),
        "valid": make_ds(valid_seqs),
    })

    output.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output}")
    dataset_dict.save_to_disk(str(output))

    loaded = load_from_disk(str(output))
    print(f"Verified: {loaded}")
    print("Done!")


if __name__ == "__main__":
    main()
