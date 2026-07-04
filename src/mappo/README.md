# MindSafe — MAPPO training package

Multi-agent PPO (proposal §7.2) that trains LoRA adapters for the three policy
agents — coordinator, therapist, monitor — to suppress the unsafe-consensus
failure mode. It reuses the frozen baseline (external judge, MAS eval harness,
patient simulator, consensus metric) and adds its own trainable policies.

## Objective

Maximize a joint reward that trades therapeutic utility against the
unsafe-consensus penalty, with a single (centralized) advantage shared across
the three agents:

```
r_joint     = r_therapy − β · c_consensus                    (Eq. 3)
c_consensus = exp(−d(h_therapist, h_monitor) / τ) · (1 − σ)  (Eq. 4)
```

The penalty is high when the agents agree in latent space (small distance `d`)
while the judge rates the turn unsafe (small `σ`).

## Modules

| Module | Role |
|---|---|
| `policy.py`    | LoRA-adapter policies; `generate` (rollout) + `compute_log_probs` (PPO ratio). |
| `reward.py`    | `c_consensus` (Eq. 4), `r_therapy`, `r_joint` (Eq. 3). |
| `value_net.py` | Centralized value head `V(s)` for GAE. |
| `rollout.py`   | `RolloutBuffer`, `collect_rollouts`, GAE. |
| `trainer.py`   | `MAPPOTrainer` — clipped-PPO update loop, checkpoints. |
| `eval.py`      | `evaluate_against_baseline` — periodic baseline-comparable eval. |

## Run

```
python scripts/train_mappo.py --config configs/mappo_4gpu.yaml --output-dir data/results/mappo/main
```

Reproduce commands are in the top-level `README.md`; reported numbers use
checkpoint u24 of `data/results/mappo/main/`.
