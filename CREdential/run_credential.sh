#!/bin/bash

#Submit this script with: sbatch --array [1-15] --output=../bash_out/output_%A_%a.out --error=../bash_out/error_%A_%a.out run_barry.sh ${SLURM_ARRAY_TASK_ID}

#SBATCH --partition=nhmem
#SBATCH --cpus-per-task=48
#SBATCH --job-name=credential
#SBATCH --mem=2000G
#SBATCH -n 1
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jake.cornwallscoones@crick.ac.uk
#SBATCH --output=bash_out.out
#SBATCH --error=bash_out.err

mkdir -p bash_out

eval "$(conda shell.bash hook)"
source activate regression_modelling

python pipeline_synergy.py