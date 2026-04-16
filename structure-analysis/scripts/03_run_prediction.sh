#!/bin/bash
# Script 3: Run AlphaFold prediction
# This runs prediction for a single target

set -e

# Arguments
TARGET_FILE=$1
TCRDOCK_PATH=$2
AF_DATA_DIR=$3
USER_OUTPUTS_DIR=$4

if [ -z "$TARGET_FILE" ] || [ -z "$TCRDOCK_PATH" ] || [ -z "$AF_DATA_DIR" ] || [ -z "$USER_OUTPUTS_DIR" ]; then
    echo "Usage: $0 <target_file> <tcrdock_path> <alphafold_data_dir> <user_outputs_dir>"
    exit 1
fi

if [ ! -f "$TARGET_FILE" ]; then
    echo "Target file not found: $TARGET_FILE"
    exit 1
fi

# Determine run ID from containing directory name
RUN_ID=$(basename "$(dirname "$TARGET_FILE")")

# Predictions directory is ALWAYS under the global work directory,
# never infer it by rewriting $USER_OUTPUTS_DIR
PRED_DIR="${PREDICTIONS_DIR}/${RUN_ID}"
mkdir -p "$PRED_DIR"

echo "Running prediction for RUN_ID: $RUN_ID"
echo "Output directory: $PRED_DIR"

python "$TCRDOCK_PATH/run_prediction.py" \
    --targets "$TARGET_FILE" \
    --outfile_prefix "$PRED_DIR/${RUN_ID}_run" \
    --model_names model_2_ptm \
    --data_dir "$AF_DATA_DIR"

echo "Prediction complete for RUN_ID: $RUN_ID"
