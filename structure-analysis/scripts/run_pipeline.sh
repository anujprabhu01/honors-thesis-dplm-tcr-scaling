#!/bin/bash
# Master Pipeline Script
# This script orchestrates the entire TCR structure prediction pipeline

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ ! -f "$SCRIPT_DIR/00_config.sh" ]; then
    echo "ERROR: Configuration file not found: $SCRIPT_DIR/00_config.sh"
    exit 1
fi

source "$SCRIPT_DIR/00_config.sh"

# Validate configuration
if ! validate_config; then
    exit 1
fi

# Create all necessary directories
mkdir -p "$WORK_DIR"
mkdir -p "$TARGETS_DIR"
mkdir -p "$USER_OUTPUTS_DIR"
mkdir -p "$PREDICTIONS_DIR"
mkdir -p "$RELAXED_DIR"
mkdir -p "$RELABELED_DIR"
mkdir -p "$INTERFACE_SCORES_DIR"
mkdir -p "$INTERFACE_LOGS_DIR"
mkdir -p "$SLURM_LOGS_DIR"

echo "========================================"
echo "TCR Structure Prediction Pipeline"
echo "========================================"
echo "Input CSV: $INPUT_CSV"
echo "Work Directory: $WORK_DIR"
echo "========================================"

###############################################################################
# STEP 1: Generate target TSV files
###############################################################################

echo ""
echo "Step 1: Generating target TSV files..."

python "$GENERATE_TARGETS_SCRIPT" \
    --input_csv "$INPUT_CSV" \
    --output_dir "$TARGETS_DIR" \
    --organism "$ORGANISM" \
    --mhc_class "$MHC_CLASS"

# Count valid targets
NUM_TARGETS=$(find "$TARGETS_DIR" -name "*.tsv" | wc -l)
echo "Generated $NUM_TARGETS target files"

if [ $NUM_TARGETS -eq 0 ]; then
    echo "ERROR: No valid targets generated"
    exit 1
fi

###############################################################################
# STEP 2: Submit setup jobs
###############################################################################

echo ""
echo "Step 2: Submitting AlphaFold setup jobs..."

SETUP_JOB_ID=$(sbatch --parsable \
    --job-name=tcr_setup \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --qos="$SLURM_QOS" \
    --array=0-$((NUM_TARGETS-1)) \
    --ntasks=1 \
    --cpus-per-task="$SETUP_CPUS" \
    --mem="$SETUP_MEM" \
    --time="$SETUP_TIME" \
    --output="$SLURM_LOGS_DIR/setup_%A_%a.out" \
    --error="$SLURM_LOGS_DIR/setup_%A_%a.err" \
    --export=ALL \
    --wrap="
        source $SCRIPT_DIR/00_config.sh
        load_conda_env
        
        TSV_FILES=(\$(find $TARGETS_DIR -name '*.tsv' | sort))
        TSV_FILE=\${TSV_FILES[\$SLURM_ARRAY_TASK_ID]}
        
        bash $SETUP_SCRIPT \"\$TSV_FILE\" \"$USER_OUTPUTS_DIR\" \"$TCRDOCK_PATH\"
    ")

echo "Submitted setup job: $SETUP_JOB_ID"

###############################################################################
# STEP 3: Submit prediction jobs (depends on setup)
###############################################################################

echo ""
echo "Step 3: Submitting AlphaFold prediction jobs..."

PREDICT_JOB_ID=$(sbatch --parsable \
    --job-name=tcr_predict \
    --dependency=afterok:$SETUP_JOB_ID \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --qos="$SLURM_QOS" \
    --gres=gpu:"$PREDICT_GPU" \
    --array=0-$((NUM_TARGETS-1)) \
    --ntasks=1 \
    --cpus-per-task="$PREDICT_CPUS" \
    --mem="$PREDICT_MEM" \
    --time="$PREDICT_TIME" \
    --output="$SLURM_LOGS_DIR/predict_%A_%a.out" \
    --error="$SLURM_LOGS_DIR/predict_%A_%a.err" \
    --export=ALL \
    --wrap="
        source $SCRIPT_DIR/00_config.sh
        load_conda_env
        load_cuda

        # SAFELY list the real target files only:
        # user_outputs/<RUN_ID>/targets.tsv
        TARGET_FILES=(\$(find \"$USER_OUTPUTS_DIR\" -mindepth 2 -maxdepth 2 -name 'targets.tsv' | sort))

        # Select correct target file by SLURM array index
        TARGET_FILE=\${TARGET_FILES[\$SLURM_ARRAY_TASK_ID]}

        echo \"Predict job running on TARGET_FILE: \$TARGET_FILE\"

        # Run prediction script
        bash $PREDICT_SCRIPT \"\$TARGET_FILE\" \"$TCRDOCK_PATH\" \"$AF_DATA_DIR\" \"$USER_OUTPUTS_DIR\"
    ")

echo "Submitted prediction job: $PREDICT_JOB_ID (depends on $SETUP_JOB_ID)"

###############################################################################
# STEP 4: Submit relaxation jobs (depends on prediction)
###############################################################################

echo ""
echo "Step 4: Submitting Rosetta relaxation jobs..."

# Note: We'll determine the number of PDB files after prediction completes
# For now, we'll use a reasonable upper bound and let tasks exit if no file exists

RELAX_JOB_ID=$(sbatch --parsable \
    --job-name=tcr_relax \
    --dependency=afterok:$PREDICT_JOB_ID \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --qos="$SLURM_QOS" \
    --array=0-$((NUM_TARGETS*10-1)) \
    --ntasks=1 \
    --cpus-per-task="$RELAX_CPUS" \
    --mem="$RELAX_MEM" \
    --time="$RELAX_TIME" \
    --output="$SLURM_LOGS_DIR/relax_%A_%a.out" \
    --error="$SLURM_LOGS_DIR/relax_%A_%a.err" \
    --export=ALL \
    --wrap="
        source $SCRIPT_DIR/00_config.sh
        
        PDB_FILES=(\$(find $PREDICTIONS_DIR -name '*.pdb' | sort))
        
        if [ \$SLURM_ARRAY_TASK_ID -ge \${#PDB_FILES[@]} ]; then
            echo 'No PDB file for task \$SLURM_ARRAY_TASK_ID'
            exit 0
        fi
        
        PDB_FILE=\${PDB_FILES[\$SLURM_ARRAY_TASK_ID]}
        
        bash $RELAX_SCRIPT \"\$PDB_FILE\" \"$ROSETTA_BIN\" \"$RELAXED_DIR\"
    ")

echo "Submitted relaxation job: $RELAX_JOB_ID (depends on $PREDICT_JOB_ID)"

###############################################################################
# STEP 5: Submit relabeling jobs (depends on relaxation)
###############################################################################

echo ""
echo "Step 5: Submitting chain relabeling jobs..."

RELABEL_JOB_ID=$(sbatch --parsable \
    --job-name=tcr_relabel \
    --dependency=afterok:$RELAX_JOB_ID \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --qos="$SLURM_QOS" \
    --array=0-$((NUM_TARGETS*10-1)) \
    --ntasks=1 \
    --cpus-per-task="$RELABEL_CPUS" \
    --mem="$RELABEL_MEM" \
    --time="$RELABEL_TIME" \
    --output="$SLURM_LOGS_DIR/relabel_%A_%a.out" \
    --error="$SLURM_LOGS_DIR/relabel_%A_%a.err" \
    --export=ALL \
    --wrap="
        source $SCRIPT_DIR/00_config.sh
        load_conda_env
        
        PDB_FILES=(\$(find $RELAXED_DIR -name '*_relaxed*.pdb' | sort))
        
        if [ \$SLURM_ARRAY_TASK_ID -ge \${#PDB_FILES[@]} ]; then
            echo 'No PDB file for task \$SLURM_ARRAY_TASK_ID'
            exit 0
        fi
        
        PDB_FILE=\${PDB_FILES[\$SLURM_ARRAY_TASK_ID]}
        BASENAME=\$(basename \"\$PDB_FILE\" .pdb)
        PREFIX=\${BASENAME%%_run_*}
        
        # Find corresponding targets.tsv
        TARGETS_TSV=\"$USER_OUTPUTS_DIR/\$PREFIX/targets.tsv\"
        
        if [ ! -f \"\$TARGETS_TSV\" ]; then
            echo 'ERROR: targets.tsv not found for \$PREFIX'
            exit 1
        fi
        
        OUTPUT_FILE=\"$RELABELED_DIR/\${BASENAME}_relabeled.pdb\"
        
        python $RELABEL_SCRIPT \
            --pdb_file \"\$PDB_FILE\" \
            --targets_tsv \"\$TARGETS_TSV\" \
            --output_file \"\$OUTPUT_FILE\" \
            --verbose
    ")

echo "Submitted relabeling job: $RELABEL_JOB_ID (depends on $RELAX_JOB_ID)"

###############################################################################
# STEP 6: Submit interface analysis jobs (depends on relabeling)
###############################################################################

echo ""
echo "Step 6: Submitting interface analysis jobs..."

INTERFACE_JOB_ID=$(sbatch --parsable \
    --job-name=tcr_interface \
    --dependency=afterok:$RELABEL_JOB_ID \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --qos="$SLURM_QOS" \
    --array=0-$((NUM_TARGETS*10-1)) \
    --ntasks=1 \
    --cpus-per-task="$INTERFACE_CPUS" \
    --mem="$INTERFACE_MEM" \
    --time="$INTERFACE_TIME" \
    --output="$SLURM_LOGS_DIR/interface_%A_%a.out" \
    --error="$SLURM_LOGS_DIR/interface_%A_%a.err" \
    --export=ALL \
    --wrap="
        source $SCRIPT_DIR/00_config.sh
        
        PDB_FILES=(\$(find $RELABELED_DIR -name '*_relabeled.pdb' | sort))
        
        if [ \$SLURM_ARRAY_TASK_ID -ge \${#PDB_FILES[@]} ]; then
            echo 'No PDB file for task \$SLURM_ARRAY_TASK_ID'
            exit 0
        fi
        
        PDB_FILE=\${PDB_FILES[\$SLURM_ARRAY_TASK_ID]}
        
        bash $INTERFACE_SCRIPT \"\$PDB_FILE\" \"$ROSETTA_BIN\" \"$INTERFACE_SCORES_DIR\" \"$INTERFACE_LOGS_DIR\"
    ")

echo "Submitted interface analysis job: $INTERFACE_JOB_ID (depends on $RELABEL_JOB_ID)"

###############################################################################
# Summary
###############################################################################

echo ""
echo "========================================"
echo "Pipeline Submitted Successfully"
echo "========================================"
echo "Job Chain:"
echo "  1. Setup:     $SETUP_JOB_ID"
echo "  2. Predict:   $PREDICT_JOB_ID"
echo "  3. Relax:     $RELAX_JOB_ID"
echo "  4. Relabel:   $RELABEL_JOB_ID"
echo "  5. Interface: $INTERFACE_JOB_ID"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: $SLURM_LOGS_DIR"
echo ""
echo "Output directories:"
echo "  Targets:          $TARGETS_DIR"
echo "  User Outputs:     $USER_OUTPUTS_DIR"
echo "  Predictions:      $PREDICTIONS_DIR"
echo "  Relaxed:          $RELAXED_DIR"
echo "  Relabeled:        $RELABELED_DIR"
echo "  Interface Scores: $INTERFACE_SCORES_DIR"
echo "========================================"
