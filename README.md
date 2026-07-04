# MindSafe

Thesis code: a multi-agent framework for safer mental-health chat. 
Three agents (coordinator, therapist, monitor) respond to a simulated patient; 
an external judge scores each turn. 

Two stages:

- **Baseline** measures the *unsafe-consensus* failure mode — the agents agree, the
  judge rates the turn unsafe, but it's still released as safe — including under
  monitor-persuasion and insider-threat attacks.
- **MAPPO** trains the agents' LoRA policies with a shared safety reward to reduce it.

Runs on Snellius (4× A100: policy on GPU 0, vLLM judge on GPUs 1–3).

Tags: `baseline-v1` = baseline results only; `mappo-v1` = full thesis (baseline + MAPPO).
`main` == `mappo-v1`.

## Install

```bash
pip install -r requirements.txt
sbatch slurm/download_models.sbatch       # or directly: python3 scripts/download_models.py
```

## Run

```bash
# smoke test first — quick end-to-end check of the setup
sbatch slurm/smoke_test.job

# baseline (clean psi arm, no attack)
sbatch slurm/run_baseline_psi.job
# or directly: python scripts/run_baseline_experiment.py --config configs/experiment_4gpu.yaml --arms psi

# MAPPO training
sbatch slurm/train_mappo.job          # ablations: set CONFIG=configs/mappo_4gpu_beta3.yaml (or _unsafetyonly)
# or directly: python scripts/train_mappo.py --config configs/mappo_4gpu.yaml --output-dir data/results/mappo/main

# attack arms (baseline only)
sbatch slurm/run_baseline_persuade_just.job         # monitor-persuasion (justification-only)
sbatch slurm/run_baseline_persuade_cot.job          # monitor-persuasion (chain-of-thought)
sbatch slurm/run_baseline_compromised_monitor.job   # insider-threat (compromised monitor)
# psi + both persuasion arms in one sweep: bash slurm/submit_all_arms.sh
```

Both runners take `--config` / `--output-dir` / `--arms`; see the `slurm/*.job` files for exact flags.

## Reproduce

```bash
# baseline (tag baseline-v1)
python3 scripts/consensus_penalty.py \
    data/results/baseline/sweep_20260425_165007 data/results/baseline/run_22408247 \
    --released-only --top-k 5 --by-scenario

# MAPPO (tag mappo-v1; eval logs already committed)
python3 scripts/analyze_mappo_diagnostic.py --run-dir data/results/mappo/main
# regenerate the eval logs from checkpoints: sbatch slurm/reeval_checkpoints.job
```

MAPPO numbers use checkpoint **u24** (training stopped early — see the note in
`configs/mappo_4gpu.yaml`). Baseline details: `data/results/baseline/MANIFEST.md`.

## Layout

```
src/agents, mas, models, simulation, redteam, evaluation   baseline framework (frozen)
src/mappo/          MAPPO training — see src/mappo/README.md
scripts/            runners + analysis
slurm/              Snellius batch jobs
configs/            experiment_4gpu.yaml (baseline), mappo_4gpu*.yaml (training)
data/results/       baseline/ and mappo*/ logs, checkpoints, eval records
```
