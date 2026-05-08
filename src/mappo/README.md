# MindSafe MAPPO — Training Package

This package implements the **multi-agent PPO** training objective from
the proposal §7.2. It is layered on top of the frozen baseline
(`src.agents`, `src.mas`, `src.evaluation`) and never modifies it.

> **Status**: scaffold only. Interfaces, type hints, and docstrings are
> in place; actual training logic is `raise NotImplementedError`.
> Implement iteratively, one module at a time.

---

## Why a separate package

`src/mappo/` is *exclusively* the training-time code. The baseline
remains a frozen reference implementation that:

- Reproduces all numbers in the paper from `git tag baseline-v1`.
- Lets you run any new ablation arm (more attacks, more scenarios) by
  editing `configs/experiment_4gpu.yaml` without ever touching MAPPO.
- Provides the eval harness that MAPPO calls during training to check
  whether the trained policy actually beats the frozen baseline.

The dependency graph is **one-way**:

```
src.mappo  ──imports──►  src.evaluation.consensus_metrics
src.mappo  ──imports──►  src.simulation.PsiPatientSimulator
src.mappo  ──imports──►  src.redteam.build_hook
src.mappo  ──imports──►  src.mas.InstrumentedMAS   (eval only)
src.mappo  ──imports──►  src.models.VLLMHTTPClient (frozen judge)

src.mappo  ──does NOT import from──►  src.agents.*
                                       (we own the trainable policies)
```

---

## Math (proposal §7.2)

Given a triad of agents (coordinator, therapist, monitor) with
joint policy π = (π_coord, π_ther, π_mon), we maximize:

$$
J(\theta) = \mathbb{E}_t \big[ R^{\text{joint}}(s_t, a_t) \big]
$$

where the joint reward (Eq. 3) trades off therapeutic utility against
the unsafe-consensus penalty:

$$
R^{\text{joint}}(s_t, a_t) = r_{\text{therapy}}(s_t, a_t) - \beta \cdot c_{\text{consensus}}(s_t)
$$

The penalty (Eq. 4) is high when agents internally agree (low cosine
distance between hidden states) and the response is judged unsafe:

$$
c_{\text{consensus}}(s_t) = \exp\!\Big(- \frac{d(h_t^{\text{ther}}, h_t^{\text{mon}})}{\tau}\Big) \cdot (1 - \sigma(s_t))
$$

We optimize via the clipped PPO objective per agent (Eq. 5), with a
**centralized advantage** Â_t shared across all three agents — this is
the key MAPPO trick that makes a coordinator's bad routing decision
penalize the therapist's adapter too.

---

## Module map

| Module | Role |
|---|---|
| `policy.py` | LoRA-adapter wrappers around the shared base model. Provides `generate` (rollout) and `compute_log_probs` (PPO ratio). Three named adapters: `coordinator`, `therapist`, `monitor`. |
| `reward.py` | `c_consensus` (Eq. 4), `r_therapy` (from judge dimensions), `r_joint` (Eq. 3). Reuses cosine-distance from baseline. |
| `value_net.py` | `CentralizedValueNet` — small MLP head on top of base model's pooled hidden states. Computes V_φ(s_t) for GAE. |
| `rollout.py` | `RolloutBuffer` + `collect_rollouts` async function. Replaces the frozen-agent path of `InstrumentedMAS`. Per-turn records token ids + log probs + hidden states + reward components. `compute_advantages` does GAE. |
| `trainer.py` | `MAPPOTrainer` — main update loop. Per-agent clipped surrogate, value loss, gradient clipping, periodic checkpoint. |
| `eval.py` | `evaluate_against_baseline` — periodic eval that runs the trained policy through `src.mas.InstrumentedMAS` to produce baseline-comparable metrics. |

---

## Suggested implementation order

1. **`reward.py`** (~30 min). Pure functions, no ML. Verify:
   - `c_consensus(...)` matches `consensus_metrics.latent_cosine_distance`
     baseline scale (run on a turn from the baseline data; numbers
     should agree).
   - `r_therapy` formula chosen and documented.
   - `r_joint` returns the right component dict.

2. **`policy.py`** (~half day). The hard PEFT plumbing.
   - Get `LoRAAgentPolicy.generate` returning text + hidden.
   - Get `compute_log_probs` returning a tensor that matches manual
     log-softmax of the model's logits on a known input.
   - Sanity check by comparing `generate(temp=0.0)` to the frozen
     baseline therapist's output on the same prompt at LoRA-init
     (LoRA-init weights = 0, should match base model).

3. **`value_net.py`** (~few hours). Minimal MLP head.
   - Initialize and forward-pass on a dummy global-state string.
   - Verify the head's output is in a reasonable scale (not NaN).

4. **`rollout.py`** (~half day). Most plumbing-heavy.
   - Get a single trajectory collected end-to-end on a real patient.
   - Log a turn record and verify all fields are populated.
   - Run GAE on a hand-built tiny buffer and compare returns to manual
     calculation.

5. **`trainer.py`** (~half day). The PPO loop.
   - Get one full update cycle to run without crashing on a buffer of 4
     trajectories. Log policy_loss, value_loss, mean_c_consensus.
   - Verify the loss decreases on a degenerate buffer (all rewards=0
     except one turn — policy should learn to avoid that turn).

6. **`eval.py`** (~few hours). Bridge to baseline harness.
   - Wrap policy in shims, run InstrumentedMAS with them.
   - Verify metrics match `scripts/consensus_penalty.py` output shape.

7. **Integrate**: `scripts/train_mappo.py` runs end-to-end on
   `configs/mappo_4gpu.yaml` for a short training run. Smoke-test on
   2 episodes before committing to a full run.

Total estimate: **~3-4 dev days** to a working v0 trainer (no perf
optimization). Add another ~1 week for tuning β, τ, lr, KL coef, etc.

---

## Things explicitly NOT in scope for v0

- **Multi-node training**. Single-node 4×A100 only.
- **Trained value-head shared with policy**. Use the frozen-base + tiny
  head version (option A in `value_net.py`).
- **Adversarial co-training**. The hook system is exposed in
  `collect_rollouts` so you *can* train against attacks, but the v0
  training runs use `hook=None`. Adversarial training is a follow-up.
- **Full proposal RQ3** (therapeutic utility vs safety trade-off). v0
  reports c_consensus down + r_therapy roughly flat. RQ3 needs a
  dedicated empathy benchmark.

---

## Reproducibility

All MAPPO runs save:
- `configs/mappo_4gpu.yaml` snapshot
- Adapter checkpoints every N updates
- Per-update logs (policy losses, value loss, mean reward components)
- Periodic eval reports (same shape as baseline)

When MAPPO is mature, tag the corresponding state as `mappo-v1` and
update `data/results/baseline/MANIFEST.md` to point at the trained
policy as a comparison column.
