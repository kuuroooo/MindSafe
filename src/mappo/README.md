# MindSafe — MAPPO training

Multi-agent PPO (proposal §7.2): trains LoRA adapters for the three agents
(coordinator, therapist, monitor) with a shared safety reward
(`r_therapy − β·c_consensus`) and one centralized advantage across all three, to
cut the unsafe-consensus failure mode. Reuses the frozen baseline (judge, eval
harness, patient simulator, consensus metric).

## Modules

| file | what |
|---|---|
| `policy.py`    | LoRA policies — `generate`, `compute_log_probs`. |
| `reward.py`    | `c_consensus`, `r_therapy`, `r_joint` (proposal Eq. 3–4). |
| `value_net.py` | centralized value head for GAE. |
| `rollout.py`   | rollout collection + GAE. |
| `trainer.py`   | clipped-PPO update loop + checkpoints. |
| `eval.py`      | eval against the frozen baseline. |

## Run

```
sbatch slurm/train_mappo.job
# or: python scripts/train_mappo.py --config configs/mappo_4gpu.yaml --output-dir data/results/mappo/main
```

Results + reproduce commands: top-level `README.md`. Reported numbers use checkpoint u24.
