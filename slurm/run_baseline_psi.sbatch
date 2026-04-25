#!/bin/bash
#SBATCH --job-name=mindsafe_psi
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=logs/psi_%j.out
#SBATCH --error=logs/psi_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=k.c.j.wang@student.vu.nl

# MindSafe baseline — arm `psi` (PATIENT-psi, no attack)
#   GPU 0   : MAS triad  (LLaMA-3-8B, HF transformers)
#   GPU 1-3 : Judge      (LLaMA-3.1-70B, vLLM HTTP subprocess)

set -e

ARM=psi

echo "=== Job start: $(date) | arm=$ARM ==="
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

export CUDA_VISIBLE_DEVICES=0,1,2,3

# All three arms write into the same sweep dir so the per-arm subtrees
# stitch together into a single results root. Set SWEEP_ID once before
# submitting all three sbatch files; otherwise each job gets its own dir.
SWEEP_ID="${SWEEP_ID:-run_${SLURM_JOB_ID}}"
OUTPUT_DIR="data/results/baseline/${SWEEP_ID}"
mkdir -p "$OUTPUT_DIR" logs

cd ~/mindsafe

python scripts/run_baseline_experiment.py \
    --config configs/experiment_4gpu.yaml \
    --output-dir "$OUTPUT_DIR" \
    --arms "$ARM"

echo "=== Job end: $(date) | arm=$ARM ==="
