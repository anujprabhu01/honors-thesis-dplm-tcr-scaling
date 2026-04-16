#!/usr/bin/env python3
"""
Script 1: Generate individual target TSV files from input CSV
"""
import os
import re
import sys
import pandas as pd
import argparse

def normalize_hla(raw):
    """
    Convert VDJdb HLA strings into TCRdock format:
      HLA-A0201     -> A*02:01
      HLA-B*0801    -> B*08:01
      B0702         -> B*07:02
      HLA-A*02:01   -> A*02:01
    """
    raw = str(raw).strip()
    
    # Remove HLA- or HLA_ prefixes
    raw = re.sub(r'^(HLA[-_])', '', raw)
    
    # Already correct (A*02:01)
    if re.match(r'^[ABCE]\*\d{2}:\d{2}$', raw):
        return raw
    
    # Case: B0702 -> B*07:02
    m = re.match(r'^([ABCE])(\d{2})(\d{2})$', raw)
    if m:
        return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"
    
    # Case: A*0201 -> A*02:01
    m = re.match(r'^([ABCE])\*(\d{2})(\d{2})$', raw)
    if m:
        return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"
    
    # Case: A0201 -> A*02:01
    m = re.match(r'^([ABCE])(\d{2})(\d{2})$', raw)
    if m:
        return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"
    
    raise ValueError(f"Unrecognized HLA format: {raw}")


def main():
    parser = argparse.ArgumentParser(description='Generate target TSV files from input CSV')
    parser.add_argument('--input_csv', required=True, help='Input CSV file with TCR data')
    parser.add_argument('--output_dir', required=True, help='Output directory for TSV files')
    parser.add_argument('--organism', default='human', help='Organism (default: human)')
    parser.add_argument('--mhc_class', type=int, default=1, help='MHC class (default: 1)')
    
    args = parser.parse_args()
    
    required_cols = {
        'Peptide', 'HLA', 'Va', 'Ja', 'CDR3a',
        'Vb', 'Jb', 'CDR3b'
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input_csv)
    
    valid_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        if any(pd.isna(row[col]) or str(row[col]).strip()=="" for col in required_cols):
            print(f"Skipping row {idx} - missing fields")
            skipped_count += 1
            continue
        
        try:
            normalized_hla = normalize_hla(row['HLA'])
        except Exception as e:
            print(f"Skipping row {idx}: Bad HLA -> {row['HLA']} ({e})")
            skipped_count += 1
            continue
        
        target = pd.DataFrame([{
            "organism": args.organism,
            "mhc_class": args.mhc_class,
            "mhc": normalized_hla,
            "peptide": row["Peptide"],
            "va": row["Va"],
            "ja": row["Ja"],
            "cdr3a": row["CDR3a"],
            "vb": row["Vb"],
            "jb": row["Jb"],
            "cdr3b": row["CDR3b"],
        }])
        
        outpath = os.path.join(args.output_dir, f"{idx}.tsv")
        target.to_csv(outpath, sep="\t", index=False)
        valid_count += 1
    
    print(f"\nSummary:")
    print(f"  Valid targets created: {valid_count}")
    print(f"  Rows skipped: {skipped_count}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
