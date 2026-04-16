# TCR Structure Prediction Pipeline

An end-to-end automated pipeline for TCR-pMHC structure prediction using TCRdock and Rosetta.

## Overview

This pipeline takes a CSV file with TCR information and automatically:
1. Generates target files for each TCR-pMHC complex
2. Sets up AlphaFold inputs
3. Predicts structures using TCRdock (modified AlphaFold pipeline)
4. Relaxes structures with Rosetta
5. Relabels chains for InterfaceAnalyzer (MHC+Peptide→A, TCRα+TCRβ→B)
6. Analyzes interfaces with Rosetta InterfaceAnalyzer

## Requirements

### Software Dependencies
- [Environment setup](./env/)
- TCRdock (https://github.com/phbradley/TCRdock)
- AlphaFold database (complete steps in "Obtaining Genetic Databases" section [here](https://github.com/google-deepmind/alphafold))
- Rosetta [here](https://rosettacommons.org/software/download/) as a non-commercial user (version 3.14). 
Required binaries:
`relax.linuxgccrelease`
`InterfaceAnalyzer.linuxgccrelease`
  
### Input CSV Format

Your input CSV must contain the following columns:
- `Peptide`: Peptide sequence
- `HLA`: HLA allele (formats supported: HLA-A0201, A*02:01, A0201, etc.)
- `Va`: TCR alpha V gene
- `Ja`: TCR alpha J gene
- `CDR3a`: TCR alpha CDR3 sequence
- `Vb`: TCR beta V gene
- `Jb`: TCR beta J gene
- `CDR3b`: TCR beta CDR3 sequence

Example:
```csv
Peptide,HLA,Va,Ja,CDR3a,Vb,Jb,CDR3b
GILGFVFTL,HLA-A*02:01,TRAV12-1,TRAJ33,CVVNGGFGNVLHC,TRBV7-2,TRBJ2-7,CASSLAPGTGELFF
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/gloriagrama/tcr_structure.git
cd tcr_structure
```

2. Setup environment [see here](./env/)

3. Ensure all scripts are executable:
```bash
chmod +x *.sh
```

## Configuration

Edit `00_config.sh` to set your system-specific paths:

### Required Settings

```bash
# Input data
export INPUT_CSV="/path/to/your/input.csv"

# Working directory (all outputs will go here)
export WORK_DIR="/path/to/your/working_directory"

# Software paths
export TCRDOCK_PATH="/path/to/TCRdock"
export AF_DATA_DIR="/path/to/alphafold_data"
export ROSETTA_BIN="/path/to/rosetta/bin"

# SLURM settings
export SLURM_ACCOUNT="your_account"
export SLURM_PARTITION="htc"
export SLURM_QOS="public"

# Conda environment
export CONDA_ENV_PATH="/path/to/your/conda/env"
```

### Optional Settings

You can also adjust:
- CPU and memory allocations for each step
- Time limits
- Organism and MHC class
- GPU requirements

## Usage

### Quick Start

1. Edit `00_config.sh` with your paths
2. Run the pipeline:
```bash
tmux new -s tcrpipe
bash run_pipeline.sh
```

That's it! The script will:
- Validate your configuration
- Generate all target files
- Submit all jobs to SLURM with proper dependencies
- Handle the entire workflow automatically

### Monitoring Progress

Monitor your jobs:
```bash
squeue -u $USER
```

Check logs:
```bash
# Setup logs
ls $WORK_DIR/slurm_logs/setup_*.out

# Prediction logs
ls $WORK_DIR/slurm_logs/predict_*.out

# Relaxation logs
ls $WORK_DIR/slurm_logs/relax_*.out

# Relabeling logs
ls $WORK_DIR/slurm_logs/relabel_*.out

# Interface analysis logs
ls $WORK_DIR/slurm_logs/interface_*.out
```

## Output Structure

```
$WORK_DIR/
├── targets/              # Individual target TSV files (one per TCR)
├── user_outputs/         # AlphaFold setup outputs
├── predictions/          # AlphaFold predicted structures
├── relaxed/             # Rosetta-relaxed structures
├── relabeled/           # Chain-relabeled structures (A/B format)
├── interface_scores/     # Interface analysis scores
├── interface_logs/       # Interface analysis logs
└── slurm_logs/          # SLURM job logs
```

## Pipeline Steps

### Step 1: Generate Targets
- Script: `01_generate_targets.py`
- Input: CSV file with TCR data
- Output: Individual TSV files in `targets/`
- Features: HLA normalization, validation, error handling

### Step 2: AlphaFold Setup
- Script: `02_setup_alphafold.sh`
- Input: Target TSV files
- Output: AlphaFold-ready inputs in `user_outputs/`
- Uses TCRdock's `setup_for_alphafold.py`

### Step 3: Structure Prediction
- Script: `03_run_prediction.sh`
- Input: Setup outputs
- Output: Predicted PDB structures in `predictions/`
- Uses AlphaFold via TCRdock's `run_prediction.py`
- Requires GPU

### Step 4: Structure Relaxation
- Script: `04_relax_structure.sh`
- Input: Predicted structures
- Output: Relaxed structures in `relaxed/`
- Uses Rosetta relax protocol

### Step 5: Chain Relabeling
- Script: `04b_relabel_chains.py`
- Input: Relaxed structures
- Output: Relabeled structures in `relabeled/`
- Purpose: Converts AlphaFold chain labels to A/B format for InterfaceAnalyzer
  - Chain A: MHC + Peptide
  - Chain B: TCRα + TCRβ

### Step 6: Interface Analysis
- Script: `05_interface_analysis.sh`
- Input: Relabeled structures
- Output: Interface metrics in `interface_scores/`
- Uses Rosetta InterfaceAnalyzer

## Troubleshooting

### Configuration Errors
If you see validation errors, check:
- All paths exist and are correct
- You have read/write permissions
- Software is properly installed

### Job Failures
Check the SLURM logs in `$WORK_DIR/slurm_logs/`:
- Look for error messages
- Verify resource allocations are sufficient
- Check that dependencies are properly installed

### Missing Dependencies
Ensure all required software is installed:
```bash
# Check Python
python --version

# Check Python packages
python -c "import pandas"
python -c "from Bio.PDB import PDBParser"

# Check if Rosetta binaries exist
ls $ROSETTA_BIN/relax.linuxgccrelease
ls $ROSETTA_BIN/InterfaceAnalyzer.linuxgccrelease

# Check TCRdock
ls $TCRDOCK_PATH/setup_for_alphafold.py
ls $TCRDOCK_PATH/run_prediction.py
```

### Common Issues

1. **"No valid targets generated"**
   - Check your input CSV format
   - Ensure required columns are present
   - Look for data validation errors in output

2. **GPU allocation failed**
   - Adjust `PREDICT_GPU` in config
   - Check GPU availability: `sinfo -o "%20P %5a %10l %6D %6t %10G"`

3. **Rosetta crashes**
   - Verify Rosetta binary path
   - Check if structures have proper format
   - Review relaxation/interface logs

4. **Relabeling fails**
   - Ensure BioPython is installed: `pip install biopython`
   - Check that targets.tsv exists for each structure
   - Verify relaxed PDB files are valid

## Resource Recommendations

Based on typical usage:
- **Setup**: 4 CPUs, 16GB RAM, 2 min
- **Prediction**: 4 CPUs, 32GB RAM, 1 GPU, 5 min
- **Relaxation**: 4 CPUs, 16GB RAM, 5 min
- **Relabeling**: 2 CPUs, 8GB RAM, 3 min
- **Interface**: 4 CPUs, 32GB RAM, 5 min

Total pipeline time: ~20 minutes per tcr

Adjust in `00_config.sh` based on your system and data.

## Advanced Usage

### Running Individual Steps

You can run individual scripts manually:

```bash
# Step 1
python 01_generate_targets.py \
    --input_csv input.csv \
    --output_dir targets/

# Step 2
bash 02_setup_alphafold.sh \
    targets/0.tsv \
    user_outputs/ \
    /path/to/TCRdock

# Step 3
bash 03_run_prediction.sh \
    user_outputs/0/targets.tsv \
    /path/to/TCRdock \
    /path/to/alphafold_data \
    user_outputs

# Step 4
bash 04_relax_structure.sh \
    predictions/0_run_model_2_ptm.pdb \
    /path/to/rosetta/bin \
    relaxed/

# Step 5
python 04b_relabel_chains.py \
    --pdb_file relaxed/0_run_model_2_ptm_relaxed.pdb \
    --targets_tsv user_outputs/0/targets.tsv \
    --output_file relabeled/0_run_model_2_ptm_relaxed_relabeled.pdb

# Step 6
bash 05_interface_analysis.sh \
    relabeled/0_run_model_2_ptm_relaxed_relabeled.pdb \
    /path/to/rosetta/bin \
    interface_scores/ \
    interface_logs/
```

### Custom Modifications

To customize the pipeline:
1. Modify individual scripts (01-05) for specific needs
2. Adjust resource allocations in `00_config.sh`
3. Change SLURM parameters in `run_pipeline.sh`

## Support

For issues or questions:
- Check the troubleshooting section
- Review SLURM logs for error messages
- Verify all dependencies are properly installed

