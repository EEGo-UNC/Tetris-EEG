#!/bin/bash
#SBATCH --job-name=eego_cpu
#SBATCH -p small
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH -t 01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

LOGDIR="${SCRATCH:-$SUBMIT_DIR}/logs"
mkdir -p "$LOGDIR"

echo "PWD: $(pwd)"
echo "LOGDIR: $LOGDIR"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-}"

module purge
module add python/3.13

source "$SUBMIT_DIR/venv/bin/activate"


python -u dreamer_models/EEGo_models.py