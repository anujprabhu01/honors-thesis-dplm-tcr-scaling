#!/bin/bash
#SBATCH --job-name=tcr_relax_v2
#SBATCH --account=grp_hlee314
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-439
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/adprabh1/tcr_predictions_output/slurm_logs/relax_v2_%A_%a.out
#SBATCH --error=/scratch/adprabh1/tcr_predictions_output/slurm_logs/relax_v2_%A_%a.err

source /scratch/adprabh1/tcr_structure/scripts/00_config.sh

PDB_FILES=($(find /scratch/adprabh1/tcr_predictions_output/predictions -name '*_model_*.pdb' | sort))

if [ $SLURM_ARRAY_TASK_ID -ge ${#PDB_FILES[@]} ]; then
    echo "No PDB file for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

PDB_FILE=${PDB_FILES[$SLURM_ARRAY_TASK_ID]}

bash $RELAX_SCRIPT "$PDB_FILE" "$ROSETTA_BIN" "/scratch/adprabh1/tcr_predictions_output/relaxed"



