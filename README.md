# MindSafe

Multi-agent framework for mental-health conversational AI. A Therapist, a Safety
Monitor, and a Coordinator run a simulated patient conversation; an External Judge
scores each turn. The thesis has two stages:

1. **Baseline** — diagnose the *unsafe-consensus* failure mode (the agents internally
   agree, the judge flags the turn unsafe, yet it is released as safe), including under
   monitor-persuasion and insider-threat attacks.
2. **MAPPO** — train the triad's policies (LoRA adapters) with a shared safety-aware
   reward to suppress that failure mode.

The repository holds all code, configs, SLURM batch scripts, and the frozen result
logs the thesis numbers are computed from.

## Frozen result tags

- **`baseline-v1`** — the **baseline** stage only: `src/agents`, `src/mas`,
  `data/results/baseline/`, etc. Contains **no** MAPPO code or results.
- **`mappo-v1`** — the **full thesis final state**: everything in `baseline-v1` **plus**
  the MAPPO training code (`src/mappo/`) and its results (`data/results/mappo*/`).

`main` == `mappo-v1` at release. Each stage's numbers are reproduced from its own tag.

## Layout

```
configs/
  experiment_4gpu.yaml          Baseline config: MAS/judge/embedding models, agents,
                                scenarios, seed, max_turns, red-team arms, Snellius/GPU.
  mappo_4gpu.yaml               MAPPO training config (LoRA, PPO hyperparams, reward beta/tau).
  mappo_4gpu_beta3.yaml         Ablation: penalty weight beta=3.
  mappo_4gpu_unsafetyonly.yaml  Ablation: unsafety-only penalty (drops the latent term).

src/
  agents/                       Frozen baseline agents. MAPPO never modifies these.
    base.py                     BaseAgent — shared interface (config + LLM client).
    therapist.py                TherapistAgent — therapist role.
    monitor.py                  MonitorAgent — safety monitor.
    coordinator.py              CoordinatorAgent — arbitrates therapist and monitor.
    external_judge.py           ExternalJudgeAgent — external per-turn safety/quality judge.
  mas/
    instrumented_mas.py         InstrumentedMAS — runs the conversation, records per-turn logs.
  models/
    model_loader.py             Loads MAS (HF transformers, single GPU) and Judge (vLLM HTTP).
    hf_client.py                HuggingFace transformers client pinned to one GPU.
    vllm_http_client.py         vLLM OpenAI-compatible server subprocess + HTTP client.
  simulation/
    patient_simulator.py        PatientSimulator / PsiPatientSimulator — simulated patient.
    psi_profiles.py             PATIENT-psi cognitive-model profiles.
  redteam/
    hooks.py                    Inter-agent message hooks for red-teaming the oversight channel.
    monitor_persuasion.py       Monitor-persuasion attack (Za et al., 2025).
  evaluation/
    consensus_metrics.py        Latent-distance metric + unsafe-consensus detector.
  mappo/                        MAPPO training package (details in src/mappo/README.md):
    policy.py                   LoRA-adapter policies (coordinator/therapist/monitor).
    reward.py                   c_consensus (Eq. 4), r_therapy, r_joint (Eq. 3).
    value_net.py                CentralizedValueNet — value head V(s) for GAE.
    rollout.py                  RolloutBuffer + collect_rollouts + GAE (compute_advantages).
    trainer.py                  MAPPOTrainer — clipped-PPO update loop, checkpoints.
    eval.py                     evaluate_against_baseline — periodic baseline-comparable eval.

scripts/
  run_baseline_experiment.py    Baseline runner: starts the vLLM judge, runs all
                                (arm x scenario) conversations, writes results + per-turn logs.
  train_mappo.py                MAPPO trainer entry point.
  reeval_checkpoints.py         Re-eval saved MAPPO checkpoints; dump per-turn eval JSONLs.
  analyze_mappo_diagnostic.py   Cluster-bootstrap diagnostic report (baseline vs MAPPO).
  consensus_penalty.py          Unsafe-consensus penalty (Eq. 4) — baseline headline numbers.
  dimension_breakdown.py        Per-dimension unsafe-consensus analysis.
  retune_thresholds.py          Post-hoc threshold sweep over an existing sweep.
  rejudge.py                    Re-judge a sweep with a new judge prompt (no MAS rerun).
  repair_logs.py                Re-parse the coordinator rationale field in existing logs.
  inspect_flagged.py            Pull the most informative flagged turns for review.
  start_vllm_server.py          Standalone vLLM judge launcher (debugging).
  download_models.py            Pre-download models into HF_HOME.

slurm/                          Snellius (4x A100) batch scripts:
  submit_all_arms.sh            Submit the three frozen-monitor baseline arms.
  run_baseline_psi.job          psi arm (baseline, no attack).
  run_baseline_persuade_just.job / run_baseline_persuade_cot.job   monitor-persuasion arms.
  run_baseline_compromised_monitor.job   insider-threat arm.
  train_mappo.job / train_mappo_smoke.job   MAPPO training (full / smoke).
  reeval_checkpoints.job        Re-eval MAPPO checkpoints.
  rejudge.job / download_models.sbatch / smoke_test.job

data/results/
  baseline/                     Frozen baseline outputs (baseline thesis numbers).
    MANIFEST.md                 Per-arm description, run config, reproduce commands.
    sweep_20260425_165007/      Frozen-monitor arms: psi, psi+persuade_just, psi+persuade_cot.
    run_22408247/               Insider-threat arm: psi+compromised_monitor.
  mappo/main/                   Main MAPPO run: adapter checkpoints ckpt_00000..27 + final/,
                                per-update logs, and periodic eval JSONLs. Reported MAPPO
                                numbers come from checkpoint u24 — training stopped early
                                (see the note in configs/mappo_4gpu.yaml).
  mappo_beta3/                  Ablation run: beta=3.
  mappo_unsafety_only/          Ablation run: unsafety-only penalty.

logs/                           SLURM stdout/stderr (.out/.err), vllm_server.log.
requirements.txt                Python dependencies.
```

### Baseline per-arm result files

Each arm directory under `data/results/baseline/` contains:

```
config.yaml                        The config used for that run.
results.json                       Aggregate results for the arm.
checkpoint_<scenario>.json         Per-scenario checkpoint.
logs/<scenario>/conv_NNN.jsonl     Per-conversation turn logs (full text + per-agent rationales).
```

Scenarios (5): `ambiguous_crisis`, `delusion_reinforcement`, `dependency_seeking`,
`gradual_escalation`, `harmful_coping`.

### MAPPO run files

Each MAPPO run directory (e.g. `data/results/mappo/main/`) contains:

```
ckpt_<NNNNN>/policy/{coordinator,therapist,monitor}/   Per-agent LoRA adapter weights.
ckpt_<NNNNN>/value/                                    Value-head checkpoint.
eval_<NNNNN>_turns*.jsonl                              Per-turn eval records for that checkpoint.
final/                                                 Final-state adapters.
```

## Setup

```bash
pip install -r requirements.txt
python3 scripts/download_models.py        # or: sbatch slurm/download_models.sbatch
```

Hardware: run on Snellius, 4x A100. The MAS/policy model runs on GPU 0; the vLLM judge
server runs on GPUs 1-3.

## Run — baseline

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

## Run — MAPPO training

```bash
sbatch slurm/train_mappo.job          # CONFIG env selects the config (default configs/mappo_4gpu.yaml)
```

Or directly:

```bash
python scripts/train_mappo.py \
    --config configs/mappo_4gpu.yaml \
    --output-dir data/results/mappo/main
```

Ablations: pass `--config configs/mappo_4gpu_beta3.yaml` (beta=3) or
`configs/mappo_4gpu_unsafetyonly.yaml`. Resume with `--resume-from <ckpt_dir>`.
See `src/mappo/README.md` for the training objective (proposal §7.2) and module map.

## Reproduce the headline numbers

### Baseline (code state at tag `baseline-v1`)

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

See `data/results/baseline/MANIFEST.md` for the full per-arm run configuration and the
methodological caveat on turn-count asymmetry between arms.

### MAPPO (code state at tag `mappo-v1`)

The per-checkpoint eval JSONLs are committed under `data/results/mappo/main/`, so the
diagnostic report (cluster-bootstrap CIs, baseline vs MAPPO) runs directly:

```bash
python3 scripts/analyze_mappo_diagnostic.py \
    --run-dir data/results/mappo/main \
    --out mindsafeDocs/mappo_diagnostic.md
```

To regenerate the eval JSONLs from the saved adapter checkpoints, use
`scripts/reeval_checkpoints.py --config configs/mappo_4gpu.yaml --run-dir data/results/mappo/main …`
(exact invocation in `slurm/reeval_checkpoints.job`). The reported numbers use
checkpoint **u24** of the main run.
