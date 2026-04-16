#!/bin/bash
# Master Pipeline Configuration
# USER: Edit this file to set your paths and parameters

###############################################################################
# REQUIRED: User must set these paths
###############################################################################

# Input CSV file containing TCR data (required columns: Peptide, HLA, Va, Ja, CDR3a, Vb, Jb, CDR3b)
export INPUT_CSV="/scratch/adprabh1/tcr_structure/data/curry_hlaA_02_01_test.csv"

# Main working directory where all outputs will be stored
export WORK_DIR="/scratch/adprabh1/tcr_predictions_output"

# TCRdock installation path
export TCRDOCK_PATH="/scratch/adprabh1/TCRdock"

# AlphaFold data directory
export AF_DATA_DIR="/scratch/adprabh1/alphafold_databases"

# Rosetta binary directory
export ROSETTA_BIN="/scratch/adprabh1/rosetta.binary.linux.release-371/main/source/bin"

###############################################################################
# SLURM Configuration
###############################################################################

# Account and partition settings
export SLURM_ACCOUNT="grp_hlee314"
export SLURM_PARTITION="htc"
export SLURM_QOS="public"

# Resource allocation for each step
export SETUP_CPUS=4
export SETUP_MEM="16G"
export SETUP_TIME="00:02:00"

export PREDICT_CPUS=4
export PREDICT_MEM="32G"
export PREDICT_TIME="00:30:00"
export PREDICT_GPU=1

export RELAX_CPUS=4
export RELAX_MEM="16G"
export RELAX_TIME="00:30:00"

export RELABEL_CPUS=2
export RELABEL_MEM="8G"
export RELABEL_TIME="00:03:00"

export INTERFACE_CPUS=4
export INTERFACE_MEM="32G"
export INTERFACE_TIME="00:05:00"

###############################################################################
# OPTIONAL: Pipeline parameters
###############################################################################

export ORGANISM="human"
export MHC_CLASS=1

###############################################################################
# Environment setup (modify if needed)
###############################################################################

export CONDA_ENV_PATH="/scratch/adprabh1/tcrdock_env" #mamba envs are more compatible with sol

# Function to load conda environment
load_conda_env() {
    # Clear Python paths
    unset PYTHONPATH
    unset PYTHONHOME
    
    # Directly set up the environment
    export PATH="/scratch/adprabh1/tcrdock_env/bin:$PATH"
    source /scratch/adprabh1/tcrdock_env/activate.sh
    
    # Verify
    echo "Python: $(which python)"
    python --version
}

# Function to load CUDA (for prediction step)
load_cuda() {
    module load cuda-12.4.1-gcc-12.1.0 2>/dev/null || true
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
}

###############################################################################
# Validation function
###############################################################################

validate_config() {
    local errors=0
    
    echo "Validating configuration..."
    
    if [ ! -f "$INPUT_CSV" ]; then
        echo "ERROR: INPUT_CSV not found: $INPUT_CSV"
        errors=$((errors + 1))
    fi
    
    if [ ! -d "$TCRDOCK_PATH" ]; then
        echo "ERROR: TCRDOCK_PATH not found: $TCRDOCK_PATH"
        errors=$((errors + 1))
    fi
    
    if [ ! -d "$AF_DATA_DIR" ]; then
        echo "ERROR: AF_DATA_DIR not found: $AF_DATA_DIR"
        errors=$((errors + 1))
    fi
    
    if [ ! -d "$ROSETTA_BIN" ]; then
        echo "ERROR: ROSETTA_BIN not found: $ROSETTA_BIN"
        errors=$((errors + 1))
    fi
    
    if [ $errors -gt 0 ]; then
        echo "Configuration validation failed with $errors error(s)"
        return 1
    fi
    
    echo "Configuration validated successfully"
    return 0
}

###############################################################################
# Derived paths (DO NOT EDIT)
###############################################################################

export TARGETS_DIR="$WORK_DIR/targets"
export USER_OUTPUTS_DIR="$WORK_DIR/user_outputs"
export PREDICTIONS_DIR="$WORK_DIR/predictions"
export RELAXED_DIR="$WORK_DIR/relaxed"
export RELABELED_DIR="$WORK_DIR/relabeled"
export INTERFACE_SCORES_DIR="$WORK_DIR/interface_scores"
export INTERFACE_LOGS_DIR="$WORK_DIR/interface_logs"
export SLURM_LOGS_DIR="$WORK_DIR/slurm_logs"

# Script paths
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GENERATE_TARGETS_SCRIPT="$SCRIPT_DIR/01_generate_targets.py"
export SETUP_SCRIPT="$SCRIPT_DIR/02_setup_alphafold.sh"
export PREDICT_SCRIPT="$SCRIPT_DIR/03_run_prediction.sh"
export RELAX_SCRIPT="$SCRIPT_DIR/04_relax_structure.sh"
export RELABEL_SCRIPT="$SCRIPT_DIR/04b_relabel_chains.py"
export INTERFACE_SCRIPT="$SCRIPT_DIR/05_interface_analysis.sh"
