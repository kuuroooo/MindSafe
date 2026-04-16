#!/bin/bash
#SBATCH --job-name=mindsafe_download
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

set -e

echo "=== Download start: $(date) ==="

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/mindsafe_env/bin/activate

export HF_HOME=/scratch-shared/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME" logs

# HF_TOKEN must be exported in the submitting shell, or set here.
# export HF_TOKEN=hf_...

cd ~/mindsafe
python scripts/download_models.py

echo "=== Download end: $(date) ==="
