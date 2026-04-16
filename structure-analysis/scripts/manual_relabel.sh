#!/bin/bash
#SBATCH --job-name=tcr_relabel_manual
#SBATCH --account=grp_hlee314
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-439
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --output=/scratch/adprabh1/tcr_predictions_output/slurm_logs/relabel_manual_%A_%a.out
#SBATCH --error=/scratch/adprabh1/tcr_predictions_output/slurm_logs/relabel_manual_%A_%a.err

source /scratch/adprabh1/tcr_structure/scripts/00_config.sh
load_conda_env

PDB_FILES=($(find /scratch/adprabh1/tcr_predictions_output/relaxed -name '*_relaxed*.pdb' | sort))

if [ $SLURM_ARRAY_TASK_ID -ge ${#PDB_FILES[@]} ]; then
    echo "No PDB file for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

PDB_FILE=${PDB_FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$PDB_FILE" .pdb)
PREFIX=${BASENAME%%_run_*}
TARGETS_TSV="/scratch/adprabh1/tcr_predictions_output/user_outputs/$PREFIX/targets.tsv"
OUTPUT_FILE="/scratch/adprabh1/tcr_predictions_output/relabeled/${BASENAME}_relabeled.pdb"

python $RELABEL_SCRIPT --pdb_file "$PDB_FILE" --targets_tsv "$TARGETS_TSV" --output_file "$OUTPUT_FILE" --verbose



