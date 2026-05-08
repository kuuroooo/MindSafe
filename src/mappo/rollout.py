"""Trajectory collection for MAPPO.

Replaces the frozen-agent path of `InstrumentedMAS` with the trainable
LoRA-adapter agents in `MultiAgentPolicy`. Patient simulator and
external judge are reused as-is from the baseline (they are NOT
trained).

Key differences from baseline rollout:
  - Each agent's response is sampled (with temperature) rather than
    treated as a fixed text. We record token ids + log-probs so the
    PPO update can recompute the ratio.
  - The hooks system (src.redteam) can still be plugged in for
    adversarial training arms — the trainable agents are evaluated
    under attack just like the frozen baseline was.
  - Per-turn we compute c_consensus and r_joint (reward.py) and store
    them in the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TurnRecord:
    """One step of one trajectory, per-agent fields included.

    For PPO we need: state context → action → log_prob → reward → value.
    Multi-agent: each agent has its own (action, log_prob); reward is
    centralized (shared across agents per the proposal's Eq. 5
    "By sharing the advantage A_t derived from the joint safety reward,
    we ensure that an agent is penalized even if it merely facilitated
    another agent's unsafe action.").
    """

    turn: int
    user_message: str

    # Per-agent: each agent's prompt context and emitted action
    coord_analysis_prompt: str = ""
    coord_analysis_text: str = ""
    coord_analysis_token_ids: List[int] = field(default_factory=list)
    coord_analysis_log_probs: List[float] = field(default_factory=list)

    therapist_prompt: str = ""
    therapist_response: str = ""
    therapist_token_ids: List[int] = field(default_factory=list)
    therapist_log_probs: List[float] = field(default_factory=list)
    therapist_hidden: Optional[np.ndarray] = None

    monitor_prompt: str = ""
    monitor_text: str = ""
    monitor_token_ids: List[int] = field(default_factory=list)
    monitor_log_probs: List[float] = field(default_factory=list)
    monitor_hidden: Optional[np.ndarray] = None

    coord_route_prompt: str = ""
    coord_route_text: str = ""
    coord_route_token_ids: List[int] = field(default_factory=list)
    coord_route_log_probs: List[float] = field(default_factory=list)

    # Reward components
    judge_result: Dict[str, Any] = field(default_factory=dict)
    sigma: float = 1.0
    c_consensus: float = 0.0
    r_therapy: float = 0.0
    r_joint: float = 0.0

    # Centralized value estimate V(s_t)
    value: float = 0.0

    # Done flag (last turn of an episode)
    done: bool = False


@dataclass
class RolloutBuffer:
    """Holds a batch of trajectories for one MAPPO update.

    After collection, the trainer calls `compute_advantages` to fill in
    GAE returns + advantages, then iterates over minibatches to compute
    the clipped PPO loss per agent (proposal Eq. 5).
    """

    trajectories: List[List[TurnRecord]] = field(default_factory=list)

    # Filled in by compute_advantages
    advantages: Optional[np.ndarray] = None  # shape [n_turns_total]
    returns: Optional[np.ndarray] = None     # shape [n_turns_total]

    def n_steps(self) -> int:
        return sum(len(t) for t in self.trajectories)

    def flatten_steps(self) -> List[TurnRecord]:
        return [step for traj in self.trajectories for step in traj]


async def collect_rollouts(
    policy,                  # MultiAgentPolicy
    value_net,               # CentralizedValueNet
    patient_factory,         # callable: (scenario, conv_idx) -> PsiPatientSimulator
    scenarios: List[str],
    n_episodes_per_scenario: int,
    max_turns: int,
    judge_client,            # frozen 70B judge — not trained
    consensus_metrics,       # for cosine distance
    hook=None,               # optional src.redteam.AdversaryHook
    beta: float = 1.0,
    tau: float = 0.1,
    base_seed: int = 0,
) -> RolloutBuffer:
    """Run `n_episodes_per_scenario × len(scenarios)` rollouts using the
    trainable policy. Return a buffer ready for the MAPPO update.

    Pseudocode:
      buffer = RolloutBuffer()
      for scen in scenarios:
        for ep in range(n_episodes_per_scenario):
          patient = patient_factory(scen, ep)
          history = []
          trajectory = []
          patient_msg = await patient.generate_message()
          for turn in range(max_turns):
            # 1. coord analyze (sample under coord adapter)
            coord_out = policy.coordinator.generate(...)
            # 2. therapist respond (sample, capture hidden)
            therapist_out = policy.therapist.generate(..., return_hidden=True)
            # 3. hook.transform_monitor_input
            # 4. monitor evaluate (sample, capture hidden)
            monitor_out = policy.monitor.generate(..., return_hidden=True)
            # 5. coord route (sample under coord adapter)
            route_out = policy.coordinator.generate(...)
            # 6. judge (frozen)
            judge_result = await judge_client.evaluate(...)
            sigma = judge_result["safety_score"]
            # 7. compute c_consensus + r_joint
            c = c_consensus(therapist_out.hidden, monitor_out.hidden,
                            sigma, tau)
            rew = r_joint(...)
            # 8. value estimate (concatenate state)
            v = value_net(state_t)
            # 9. record TurnRecord
            trajectory.append(TurnRecord(...))
            # 10. patient generates next message
            patient_msg = await patient.generate_message(
                assistant_response=therapist_out.text)
          buffer.trajectories.append(trajectory)
      return buffer

    TODO:
      - Hidden-state extraction from generate() — same approach as
        src.models.hf_client.HFClient.generate_with_hidden, but adapted
        to the LoRA-wrapped model. Must also record per-token log-probs
        for the PPO ratio.
      - Decide whether to train coord on both `analyze` and `route`
        prompts (yes — they're the same agent, same adapter).
      - Decide whether to allow the revision loop during training.
        Likely OFF during training (each trajectory turn is a single
        forward pass, not a multi-attempt loop) — but ON during eval.
    """
    raise NotImplementedError


def compute_advantages(
    buffer: RolloutBuffer,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> RolloutBuffer:
    """Generalized advantage estimation (GAE-λ).

    For each trajectory, compute:
        δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) − V(s_t)
        A_t = δ_t + γ·λ · A_{t+1} · (1 - done_t)
        R_t = A_t + V(s_t)         (return, used for value loss)

    The advantage is *centralized* — same A_t shared across all agents
    per proposal Eq. 5.

    Mutates `buffer.advantages` and `buffer.returns` in place; returns
    the same buffer for chaining.
    """
    raise NotImplementedError
