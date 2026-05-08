"""MAPPO update loop — clipped PPO objective per agent, shared advantage.

Implements proposal Eq. 5:

    L_CLIP_i(θ_i) = E_t[ min( ρ_t(θ_i) · Â_t,
                              clip(ρ_t(θ_i), 1−ε, 1+ε) · Â_t ) ]

where ρ_t(θ_i) = π_θ_i(a_t^i | τ_t^i) / π_θ_i_old(a_t^i | τ_t^i)
and Â_t is the *centralized* advantage shared across all agents.

The trainer:
  1. Collects rollouts (rollout.collect_rollouts) under the current policy.
  2. Computes GAE advantages (rollout.compute_advantages) using the value net.
  3. For n_epochs iterations:
     a. Splits rollout into minibatches.
     b. For each agent (coord, therapist, monitor):
        - Recompute log-probs under the current adapter.
        - Compute ratio, clipped surrogate, policy loss.
     c. Compute value loss against returns.
     d. Backprop, clip gradients, step optimizer.
  4. Logs training stats (policy loss per agent, value loss, mean
     c_consensus, mean σ).
  5. Periodically saves adapters + value net.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .policy import MultiAgentPolicy
from .rollout import RolloutBuffer
from .value_net import CentralizedValueNet


@dataclass
class MAPPOConfig:
    clip_eps: float = 0.2
    n_epochs_per_update: int = 4
    minibatch_size: int = 32
    lr_policy: float = 1e-5
    lr_value: float = 1e-4
    grad_clip: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Reward shaping
    beta: float = 1.0    # consensus penalty weight (Eq. 3)
    tau: float = 0.1     # latent-distance kernel temp (Eq. 4)


class MAPPOTrainer:
    """Clipped PPO over the three LoRA adapters + the value head.

    All three agents share a centralized advantage; their gradients
    flow through their own adapter parameters only.
    """

    def __init__(
        self,
        policy: MultiAgentPolicy,
        value_net: CentralizedValueNet,
        cfg: MAPPOConfig,
    ):
        self.policy = policy
        self.value_net = value_net
        self.cfg = cfg

        # TODO:
        #   - Build optimizer over policy.trainable_parameters() with lr_policy
        #   - Build optimizer over value_net.trainable_parameters() with lr_value
        #   - (Or one combined AdamW with parameter groups)
        self.policy_optim = None
        self.value_optim = None

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """One PPO update pass over the buffer. Returns logging dict.

        Pseudocode:
          for epoch in range(self.cfg.n_epochs_per_update):
            for batch in minibatches(buffer):
              # Per-agent policy losses (shared advantage)
              for agent_name in ("coordinator", "therapist", "monitor"):
                logp_new = self.policy.<agent>.compute_log_probs(...)
                logp_old = batch.<agent>_log_probs
                ratio = exp(logp_new - logp_old)
                surr1 = ratio * batch.advantages
                surr2 = clip(ratio, 1-eps, 1+eps) * batch.advantages
                policy_loss[agent] = -min(surr1, surr2).mean()
              # Value loss
              v_pred = self.value_net.batched(batch.global_states)
              value_loss = ((v_pred - batch.returns) ** 2).mean()
              # Optional entropy bonus per agent
              total = sum(policy_losses) + value_coef * value_loss
                      - entropy_coef * sum(entropies)
              total.backward()
              grad clip
              step
        """
        raise NotImplementedError

    def save_checkpoint(self, dir_path: Path) -> None:
        """Save policy adapters + value net head + optimizer state."""
        raise NotImplementedError

    def load_checkpoint(self, dir_path: Path) -> None:
        """Resume from a checkpoint."""
        raise NotImplementedError
