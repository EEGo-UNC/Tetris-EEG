#!/bin/bash
#SBATCH --job-name=eego_gpu
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

LOGDIR="${SCRATCH:-$SLURM_SUBMIT_DIR}/logs"
mkdir -p "$LOGDIR"

echo "PWD: $(pwd)"
echo "LOGDIR: $LOGDIR"

module purge
module load python/3.13
source venv/bin/activate

export TF_FORCE_GPU_ALLOW_GROWTH=true
python -u dreamer_models/EEGo_models.py
