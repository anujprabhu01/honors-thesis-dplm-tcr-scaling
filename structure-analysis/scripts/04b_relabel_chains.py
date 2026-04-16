#!/usr/bin/env python3
"""
Script 4b: Relabel PDB chains from AlphaFold format to A/B format
This prepares structures for Rosetta InterfaceAnalyzer
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import io
import traceback

# BioPython imports
from Bio.PDB import PDBParser, PDBIO, Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure


def clean_pdb_for_parsing(path):
    """
    Remove Rosetta junk: keep only ATOM/HETATM lines
    """
    cleaned = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                cleaned.append(line)
    return io.StringIO("".join(cleaned))


def extract_plan(tsv_path):
    """
    Extract plan for chain splitting using target_chainseq from targets.tsv
    
    Returns:
        chainA_idx: list of residue indices for chain A (MHC + peptide)
        chainB_idx: list of residue indices for chain B (TCRa + TCRb)
    """
    df = pd.read_csv(tsv_path, sep="\t")
    row = df.iloc[0]

    chain_seqs = row["target_chainseq"].split("/")
    if len(chain_seqs) != 4:
        raise ValueError(f"Expected 4 chain segments; got {len(chain_seqs)}")

    lengths = [len(s) for s in chain_seqs]

    # Compute offsets for each segment
    offsets = [0]
    for L in lengths[:-1]:
        offsets.append(offsets[-1] + L)

    mhc_start, peptide_start, tcra_start, tcrb_start = offsets

    # MHC + peptide -> chain A
    chainA = list(range(mhc_start, tcra_start))

    # TCRa + TCRb -> chain B
    chainB = list(range(tcra_start, tcrb_start + lengths[3]))

    return chainA, chainB


def relabel_pdb(input_path, output_path, chainA_idx, chainB_idx, verbose=True):
    """
    Relabel PDB chains into chain A and B for InterfaceAnalyzer
    """
    if verbose:
        print(f"  Reading PDB (cleaning Rosetta junk): {input_path}")
    
    handle = clean_pdb_for_parsing(input_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", handle)
    model = structure[0]

    chainA = Chain.Chain("A")
    chainB = Chain.Chain("B")

    flat_idx = 0

    # Iterate in correct flat order
    for ch in model:
        for res in ch:
            if flat_idx in chainA_idx:
                chainA.add(res.copy())
            elif flat_idx in chainB_idx:
                chainB.add(res.copy())
            flat_idx += 1

    if verbose:
        print(f"  Final chainA residues: {len(chainA)}")
        print(f"  Final chainB residues: {len(chainB)}")

    # Build new output structure
    new_structure = Structure("relabeled")
    new_model = Model(0)
    new_model.add(chainA)
    new_model.add(chainB)
    new_structure.add(new_model)

    io_obj = PDBIO()
    io_obj.set_structure(new_structure)
    io_obj.save(str(output_path))

    if verbose:
        print(f"  Saved: {output_path}")


def extract_prefix_from_relaxed_name(name: str) -> str:
    """
    Extract prefix from relaxed filename
    e.g. "10839_run_model_2_ptm_relaxed" -> "10839"
    """
    return name.split("_run_")[0]


def main():
    parser = argparse.ArgumentParser(description='Relabel PDB chains for InterfaceAnalyzer')
    parser.add_argument('--pdb_file', required=True, help='Input PDB file to relabel')
    parser.add_argument('--targets_tsv', required=True, help='Corresponding targets.tsv file')
    parser.add_argument('--output_file', required=True, help='Output relabeled PDB file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    pdb_path = Path(args.pdb_file)
    tsv_path = Path(args.targets_tsv)
    output_path = Path(args.output_file)
    
    if not pdb_path.exists():
        print(f"ERROR: PDB file not found: {pdb_path}")
        sys.exit(1)
    
    if not tsv_path.exists():
        print(f"ERROR: targets.tsv not found: {tsv_path}")
        sys.exit(1)
    
    if output_path.exists():
        if args.verbose:
            print(f"Output already exists: {output_path}")
        sys.exit(0)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        chainA_idx, chainB_idx = extract_plan(tsv_path)
        if args.verbose:
            print(f"Chain A residues: {len(chainA_idx)}")
            print(f"Chain B residues: {len(chainB_idx)}")
        
        relabel_pdb(pdb_path, output_path, chainA_idx, chainB_idx, verbose=args.verbose)
        
        if args.verbose:
            print("Relabeling complete")
        
    except Exception as e:
        print(f"ERROR relabeling {pdb_path.name}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
