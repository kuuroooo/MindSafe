"""MindSafe MAPPO training package.

Implements the multi-agent PPO objective from the proposal §7.2:

    R_joint(s_t, a_t) = r_therapy(s_t, a_t) - β · c_consensus(s_t)        (Eq. 3)
    c_consensus(s_t)  = exp(-d(h_therapist, h_monitor) / τ) · (1 - σ(s_t)) (Eq. 4)
    L_CLIP_i(θ_i)     = E_t[ min(ρ_t · A_t, clip(ρ_t, 1±ε) · A_t) ]       (Eq. 5)

The package is layered on top of the frozen baseline (`src.agents`,
`src.mas`, `src.evaluation`) and never modifies it. See README.md.
"""

from .policy import LoRAAgentPolicy, MultiAgentPolicy
from .reward import c_consensus, c_consensus_from_distance, r_joint, r_therapy
from .rollout import RolloutBuffer, collect_rollouts, compute_advantages
from .trainer import MAPPOConfig, MAPPOTrainer
from .value_net import CentralizedValueNet

__all__ = [
    # policy
    "LoRAAgentPolicy",
    "MultiAgentPolicy",
    # rollout
    "RolloutBuffer",
    "collect_rollouts",
    "compute_advantages",
    # trainer
    "MAPPOTrainer",
    "MAPPOConfig",
    # value net
    "CentralizedValueNet",
    # reward
    "c_consensus",
    "c_consensus_from_distance",
    "r_joint",
    "r_therapy",
]
