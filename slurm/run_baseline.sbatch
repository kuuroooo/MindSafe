#!/bin/bash
#SBATCH --job-name=mindsafe_baseline
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=k.c.j.wang@student.vu.nl

# MindSafe Baseline — 4×A100
#   GPU 0   : MAS triad (LLaMA-3-8B, HF transformers)
#   GPU 1-3 : Judge (LLaMA-3.1-70B, vLLM HTTP subprocess)

set -e

echo "=== Job start: $(date) ==="
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/mindsafe_env/bin/activate

export HF_HOME=/scratch-shared/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# All 4 GPUs visible to the parent process.
# The vLLM subprocess overrides CUDA_VISIBLE_DEVICES internally to 1,2,3.
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OUTPUT_DIR="data/results/baseline/run_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR" logs

cd ~/mindsafe

python scripts/run_baseline_experiment.py \
    --config configs/experiment_4gpu.yaml \
    --output-dir "$OUTPUT_DIR"

echo "=== Job end: $(date) ==="
