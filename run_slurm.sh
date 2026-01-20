#!/bin/bash
#SBATCH --job-name=eego_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

source ~/venv/bin/activate

export TF_FORCE_GPU_ALLOW_GROWTH=true

python -u dreamer_models/EEGo_models.py
