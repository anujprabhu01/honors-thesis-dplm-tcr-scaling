#!/usr/bin/env python
"""Prepare epitope-TCR SFT data for conditional DPLM training.

Parses text files with format EPITOPE$TCR<EOS> and saves as a
HuggingFace dataset compatible with TCRConditionalDataModule.

Usage:
    python scripts/scaling/prepare_sft_data.py

Output:
    data-bin/tcr_sft/ - HuggingFace dataset with train/valid/test splits
"""

from pathlib import Path

from datasets import Dataset, DatasetDict


# Configuration
SOURCE_DIR = '/mnt/disk11/user/xiaoyih1/data/compact_format/merged_seed42_epi_test_pair_split'
OUTPUT_DIR = Path('data-bin/tcr_sft')


def parse_file(filepath):
    """Parse a text file with lines of format EPITOPE$TCR<EOS>."""
    epitopes = []
    tcrs = []
    skipped = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Strip <EOS> suffix if present
            if line.endswith('<EOS>'):
                line = line[:-5]
            # Split on $ separator
            parts = line.split('$')
            if len(parts) != 2:
                skipped += 1
                continue
            epitope, tcr = parts
            if not epitope or not tcr:
                skipped += 1
                continue
            epitopes.append(epitope)
            tcrs.append(tcr)
    return epitopes, tcrs, skipped


def main():
    splits = {}
    for split_name, filename in [('train', 'train.txt'), ('valid', 'val.txt'), ('test', 'test.txt')]:
        filepath = Path(SOURCE_DIR) / filename
        print(f"Parsing {filepath}")
        epitopes, tcrs, skipped = parse_file(filepath)
        print(f"  {split_name}: {len(epitopes):,} pairs, {skipped} skipped")
        if epitopes:
            # length = epitope + 1 (sep token) + tcr
            lengths = [len(e) + 1 + len(t) for e, t in zip(epitopes, tcrs)]
            print(f"  Epitope lengths: {min(len(e) for e in epitopes)}-{max(len(e) for e in epitopes)}")
            print(f"  TCR lengths: {min(len(t) for t in tcrs)}-{max(len(t) for t in tcrs)}")
            splits[split_name] = Dataset.from_dict({
                'epitope': epitopes,
                'tcr': tcrs,
                'length': lengths,
            })
        else:
            splits[split_name] = Dataset.from_dict({
                'epitope': [],
                'tcr': [],
                'length': [],
            })

    dataset_dict = DatasetDict(splits)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {OUTPUT_DIR}")
    dataset_dict.save_to_disk(str(OUTPUT_DIR))

    # Verify
    print("\nVerifying saved dataset...")
    from datasets import load_from_disk
    loaded = load_from_disk(str(OUTPUT_DIR))
    print(f"Loaded dataset: {loaded}")
    for split_name in loaded:
        if len(loaded[split_name]) > 0:
            print(f"  {split_name} sample: {loaded[split_name][0]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
