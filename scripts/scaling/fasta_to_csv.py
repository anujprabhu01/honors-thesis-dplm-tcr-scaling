#!/usr/bin/env python
"""Convert generated FASTA files to a single CSV for evaluation.

Each FASTA header must contain EPITOPE=<epi> (as written by generate_conditional.py).

Usage:
    python scripts/scaling/fasta_to_csv.py --input_dir results/conditional_gen --output results/conditional_gen/sft_0.1m.csv
    python scripts/scaling/fasta_to_csv.py --input_dir results/conditional_gen  # output defaults to <input_dir>/generated.csv
"""

import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert FASTA files to CSV for evaluation")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .fasta files")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: <input_dir>/generated.csv)")
    return parser.parse_args()


def fasta_to_csv(input_dir: Path, output: Path):
    fasta_files = sorted(input_dir.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta files found in {input_dir}")

    rows = []
    for f in fasta_files:
        current_epi = None
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    m = re.search(r"EPITOPE=([^|]+)", line)
                    current_epi = m.group(1) if m else f.stem.replace("epitope_", "")
                elif line and current_epi is not None:
                    if line.isalpha():  # skip sequences with <sep>, <unk>, or other artifacts
                        rows.append({"tcr": line, "epi": current_epi})

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        fh.write("TCRs,Epitopes\n")
        for row in rows:
            fh.write(f"{row['tcr']},{row['epi']}\n")

    print(f"Written {len(rows)} rows to {output}")
    return len(rows)


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output) if args.output else input_dir / "generated.csv"
    fasta_to_csv(input_dir, output)
