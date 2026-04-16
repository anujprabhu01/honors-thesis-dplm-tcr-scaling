#!/bin/bash
# Script 4: Relax structure with Rosetta
# This relaxes a single PDB file

set -e

# Arguments
PDB_FILE=$1
ROSETTA_BIN=$2
RELAX_OUTPUT_DIR=$3

if [ -z "$PDB_FILE" ] || [ -z "$ROSETTA_BIN" ] || [ -z "$RELAX_OUTPUT_DIR" ]; then
    echo "Usage: $0 <pdb_file> <rosetta_bin_dir> <relax_output_dir>"
    exit 1
fi

if [ ! -f "$PDB_FILE" ]; then
    echo "PDB file not found: $PDB_FILE"
    exit 1
fi

mkdir -p "$RELAX_OUTPUT_DIR"

BASE=$(basename "$PDB_FILE" .pdb)

echo "Relaxing: $PDB_FILE"

"$ROSETTA_BIN/relax.static.linuxgccrelease" \
    -s "$PDB_FILE" \
    -relax:default_repeats 5 \
    -nstruct 1 \
    -out:suffix _relaxed \
    -out:path:all "$RELAX_OUTPUT_DIR" \
    > "$RELAX_OUTPUT_DIR/${BASE}_relax.log" 2>&1

echo "Relaxation complete for: $BASE"
