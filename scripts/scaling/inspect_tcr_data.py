#!/usr/bin/env python
"""Inspect the TCR repertoire pickle file.

Usage:
    python scripts/scaling/inspect_tcr_data.py
"""

import pickle

DATA_PATH = '/mnt/disk11/user/xiaoyih1/data/tcr_data_all/data/tcr_repertoires_healthy_samples/tcr_repertoire_seqs.pkl'


def main():
    print(f"Loading: {DATA_PATH}")
    with open(DATA_PATH, 'rb') as f:
        df = pickle.load(f)

    print(f"\nType: {type(df)}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nSequence length stats:")
    print(df['tcr'].str.len().describe())

    print("\nExample sequences:")
    for seq in df['tcr'].head(10):
        print(f"  {seq} (len={len(seq)})")

    print(f"\nUnique sequences: {df['tcr'].nunique()}")
    print(f"Total sequences: {len(df)}")


if __name__ == "__main__":
    main()
