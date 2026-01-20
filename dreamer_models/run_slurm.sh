#!/bin/bash
#SBATCH --job-name=eego_gpu
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Go to repo root (parent of dreamer_models/)
cd "$(dirname "$0")/.."

mkdir -p logs

module purge
module load python/3.13

# venv is at repo root
source venv/bin/activate

echo "PWD: $(pwd)"
echo "Python: $(which python)"
python --version

export TF_FORCE_GPU_ALLOW_GROWTH=true

python -u dreamer_models/EEGo_models.py
