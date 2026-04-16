#!/bin/bash
# Script 5: Run Rosetta InterfaceAnalyzer
# This analyzes the interface of a single PDB file

set -e

# Arguments
PDB_FILE=$1
ROSETTA_BIN=$2
SCORES_OUTPUT_DIR=$3
LOGS_OUTPUT_DIR=$4

if [ -z "$PDB_FILE" ] || [ -z "$ROSETTA_BIN" ] || [ -z "$SCORES_OUTPUT_DIR" ] || [ -z "$LOGS_OUTPUT_DIR" ]; then
    echo "Usage: $0 <pdb_file> <rosetta_bin_dir> <scores_output_dir> <logs_output_dir>"
    exit 1
fi

if [ ! -f "$PDB_FILE" ]; then
    echo "PDB file not found: $PDB_FILE"
    exit 1
fi

mkdir -p "$SCORES_OUTPUT_DIR" "$LOGS_OUTPUT_DIR"

BASE=$(basename "$PDB_FILE" .pdb)
SCORE_OUT="${SCORES_OUTPUT_DIR}/${BASE}_scores.sc"
LOG_OUT="${LOGS_OUTPUT_DIR}/${BASE}.log"

# Skip if already scored
if [ -f "$SCORE_OUT" ]; then
    echo "Already scored: $BASE"
    exit 0
fi

echo "Running InterfaceAnalyzer on: $PDB_FILE"

"$ROSETTA_BIN/InterfaceAnalyzer.static.linuxgccrelease" \
    -s "$PDB_FILE" \
    -interface A_B \
    -ignore_unrecognized_res \
    -pack_separated \
    -nstruct 1 \
    -out:file:scorefile "$SCORE_OUT" \
    -out:path:all "$SCORES_OUTPUT_DIR" \
    > "$LOG_OUT" 2>&1

echo "Interface analysis complete for: $BASE"
