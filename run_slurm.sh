#!/bin/bash
#SBATCH --job-name=eego_gpu
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

LOGDIR="${SCRATCH:-$SUBMIT_DIR}/logs"
mkdir -p "$LOGDIR"

echo "PWD: $(pwd)"
echo "LOGDIR: $LOGDIR"

module purge
module load python/3.13
module load cuda/12.9

source venv/bin/activate

export TF_FORCE_GPU_ALLOW_GROWTH=true
if [[ -n "${CUDA_HOME:-}" ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
fi

python -u dreamer_models/EEGo_models.py
