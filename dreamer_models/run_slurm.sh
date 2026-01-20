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

# If this script is in dreamer_models/, this puts you there:
cd "$(dirname "$0")"

# Make logs directory (either keep logs in repo root or in dreamer_models)
mkdir -p logs

module purge
module load python/3.13

# Activate venv located one level up (repo root)
source ../venv/bin/activate

echo "PWD: $(pwd)"
echo "Python: $(which python)"
python --version

export TF_FORCE_GPU_ALLOW_GROWTH=true

# Now you're already in dreamer_models, so run the script directly
python -u EEGo_models.py
