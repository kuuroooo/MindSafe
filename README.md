# MindSafe

Multi-agent framework for mental-health conversational AI. A Therapist, a Safety
Monitor, and a Coordinator run a simulated patient conversation; an External Judge
scores each turn. The repository holds the code, the experiment configuration, the
SLURM batch scripts, and the frozen result logs the thesis numbers are computed from.

The results in the thesis are frozen at git tag `baseline-v1`.

## Layout

```
configs/experiment_4gpu.yaml   Experiment config: MAS model, judge model, embedding
                               model, agent settings, scenarios, seed, max_turns,
                               red-team arms, output dir, Snellius/GPU settings.

src/
  agents/
    base.py                    BaseAgent — shared interface (config + LLM client).
    therapist.py               TherapistAgent — therapist role.
    monitor.py                 MonitorAgent — safety monitor.
    coordinator.py             CoordinatorAgent — arbitrates therapist and monitor.
    external_judge.py          ExternalJudgeAgent — external per-turn safety/quality judge.
  mas/
    instrumented_mas.py        InstrumentedMAS — runs the multi-agent conversation and
                               records per-turn logs.
  models/
    model_loader.py            Loads MAS (HF transformers, single GPU) and Judge (vLLM HTTP).
    hf_client.py               HuggingFace transformers client pinned to one GPU.
    vllm_http_client.py        vLLM OpenAI-compatible server subprocess + HTTP client.
  simulation/
    patient_simulator.py       PatientSimulator / PsiPatientSimulator — simulated patient.
    psi_profiles.py            PATIENT-psi cognitive-model profiles.
  redteam/
    hooks.py                   Inter-agent message hooks for red-teaming the oversight channel.
    monitor_persuasion.py      Monitor-persuasion attack (Za et al., 2025).
  evaluation/
    consensus_metrics.py       Consensus metrics.

scripts/
  run_baseline_experiment.py   Main experiment runner. Starts the vLLM judge server,
                               loads the MAS model, runs all (arm x scenario) conversations,
                               writes results and per-turn logs, tears down the server.
  start_vllm_server.py         Standalone vLLM judge launcher (debugging).
  download_models.py           Pre-download models into HF_HOME before running.
  consensus_penalty.py         Compute the unsafe-consensus penalty (proposal Eq. 4).
                               Produces the headline numbers.
  dimension_breakdown.py       Per-dimension unsafe-consensus analysis.
  retune_thresholds.py         Post-hoc threshold sweep over an existing sweep's records.
  rejudge.py                   Re-judge an existing sweep with a new judge prompt (no MAS rerun).
  repair_logs.py               Re-parse the coordinator rationale field in existing logs.
  inspect_flagged.py           Pull the most informative flagged turns for qualitative review.

slurm/
  submit_all_arms.sh           Submit the three frozen-monitor arms under one SWEEP_ID.
  run_baseline_psi.job         psi arm (baseline, no attack).
  run_baseline_persuade_just.job   psi+persuade_just arm.
  run_baseline_persuade_cot.job    psi+persuade_cot arm.
  run_baseline_compromised_monitor.job   psi+compromised_monitor arm (insider-threat).
  rejudge.job                  Re-judge job.
  download_models.sbatch       Model download job.
  smoke_test.job               Smoke test.

data/results/baseline/         Frozen experiment outputs. Thesis numbers computed from here.
  MANIFEST.md                  Per-arm description, run config, and reproduce commands.
  sweep_20260425_165007/       Three frozen-monitor arms: psi, psi+persuade_just, psi+persuade_cot.
  run_22408247/                Insider-threat arm: psi+compromised_monitor.

logs/                          SLURM stdout/stderr (.out/.err) from the runs, plus vllm_server.log.
requirements.txt               Python dependencies.
```

### Per-arm result files

Each arm directory under `data/results/baseline/` contains:

```
config.yaml                        The config used for that run.
results.json                       Aggregate results for the arm.
checkpoint_<scenario>.json         Per-scenario checkpoint.
logs/<scenario>/conv_NNN.jsonl     Per-conversation turn logs (full text + per-agent rationales).
```

Scenarios (5): `ambiguous_crisis`, `delusion_reinforcement`, `dependency_seeking`,
`gradual_escalation`, `harmful_coping`.

## Setup

```bash
pip install -r requirements.txt
python3 scripts/download_models.py        # or: sbatch slurm/download_models.sbatch
```

Hardware: run on Snellius, 4x A100. The MAS model runs on GPU 0; the vLLM judge
server runs on GPUs 1-3.

## Run

Frozen-monitor arms (psi, persuade_just, persuade_cot):

```bash
bash slurm/submit_all_arms.sh
```

Insider-threat arm:

```bash
sbatch slurm/run_baseline_compromised_monitor.job
```

Single arm directly (the SLURM jobs call this):

```bash
python scripts/run_baseline_experiment.py \
    --config configs/experiment_4gpu.yaml \
    --output-dir <output_dir> \
    --arms <arm>
```

Runner flags: `--config`, `--output-dir`, `--scenarios`, `--n-conversations`, `--arms`.

## Reproduce the headline numbers

From the frozen results (code state at tag `baseline-v1`):

```bash
python3 scripts/consensus_penalty.py \
    data/results/baseline/sweep_20260425_165007 \
    data/results/baseline/run_22408247 \
    --released-only --top-k 5 --by-scenario
```

Calibrated-threshold flagged-event view:

```bash
python3 scripts/retune_thresholds.py \
    data/results/baseline/sweep_20260425_165007 \
    --safety-threshold 0.7 --distance-threshold 0.07
```

See `data/results/baseline/MANIFEST.md` for the full per-arm run configuration and
the methodological caveat on turn-count asymmetry between arms.
