#!/bin/bash
# Script 2: Setup for AlphaFold prediction
# This runs setup_for_alphafold.py for a single target TSV file

set -e

# Arguments
TSV_FILE=$1
OUTPUT_DIR=$2
TCRDOCK_PATH=$3

if [ -z "$TSV_FILE" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$TCRDOCK_PATH" ]; then
    echo "Usage: $0 <tsv_file> <output_dir> <tcrdock_path>"
    exit 1
fi

TARGET_NAME=$(basename "$TSV_FILE" .tsv)
TARGET_OUTPUT_DIR="$OUTPUT_DIR/$TARGET_NAME"
TARGETS_PATH="$TARGET_OUTPUT_DIR/targets.tsv"

if [ -f "$TARGETS_PATH" ]; then
    echo "Skipping: $TARGETS_PATH already exists"
    exit 0
fi

mkdir -p "$TARGET_OUTPUT_DIR"
echo "Running setup: $TSV_FILE -> $TARGET_OUTPUT_DIR"

python "$TCRDOCK_PATH/setup_for_alphafold.py" \
    --targets_tsvfile "$TSV_FILE" \
    --output_dir "$TARGET_OUTPUT_DIR" \
    --new_docking

echo "Setup complete for: $TARGET_NAME"
