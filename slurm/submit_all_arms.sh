#!/bin/bash
# Submit the three per-arm baseline jobs with a shared SWEEP_ID so their
# outputs land under one tree: data/results/baseline/<SWEEP_ID>/<arm>/
#
# Usage:
#   bash slurm/submit_all_arms.sh                # auto SWEEP_ID from date
#   bash slurm/submit_all_arms.sh my_sweep_01    # custom SWEEP_ID

set -e

SWEEP_ID="${1:-sweep_$(date +%Y%m%d_%H%M%S)}"
export SWEEP_ID

echo "SWEEP_ID=$SWEEP_ID"
echo "Output root: data/results/baseline/$SWEEP_ID"

# --export=ALL propagates SWEEP_ID into each job's environment.
JOB_PSI=$(sbatch --parsable --export=ALL slurm/run_baseline_psi.job)
JOB_JUST=$(sbatch --parsable --export=ALL slurm/run_baseline_persuade_just.job)
JOB_COT=$(sbatch --parsable --export=ALL slurm/run_baseline_persuade_cot.job)

echo "Submitted: psi=$JOB_PSI  persuade_just=$JOB_JUST  persuade_cot=$JOB_COT"
echo "Watch with: squeue -u \$USER"
