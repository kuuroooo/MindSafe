"""Reward functions for MAPPO training (proposal §7.2 Eqs 3-4).

The signal we minimize during training is the unsafe-consensus penalty
`c_consensus`. The signal we maximize is therapeutic utility `r_therapy`.
The joint reward subtracts the penalty with a hyperparameter `β`.

These functions operate on a single turn dict — the same shape produced
by InstrumentedMAS. They're imported into rollout.py to label each step
of a trajectory.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def c_consensus(
    therapist_hidden: np.ndarray,
    monitor_hidden: np.ndarray,
    sigma: float,
    tau: float = 0.1,
) -> float:
    """Proposal Eq. 4 — unsafe-consensus cost.

        c = exp(-d(h_therapist, h_monitor) / τ) · (1 - σ)

    High when the agents' latent states agree (small d) AND the judge
    rates the response unsafe (small σ). This is the term MAPPO is
    trying to drive down.

    Reuses the same cosine distance the baseline detector uses; we
    deliberately match `src.evaluation.consensus_metrics.latent_cosine_distance`
    here so the training signal and the baseline analysis are on the
    same scale.
    """
    a = np.asarray(therapist_hidden, dtype=np.float32).reshape(-1)
    b = np.asarray(monitor_hidden, dtype=np.float32).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        d = 1.0
    else:
        sim = float(np.dot(a, b) / (na * nb))
        sim = max(-1.0, min(1.0, sim))
        d = 1.0 - sim
    return math.exp(-d / tau) * (1.0 - float(sigma))


def r_therapy(judge_result: dict, alpha_quality: float = 1.0) -> float:
    """Therapeutic utility, derived from the judge's per-dimension scores.

    Default: use the `therapeutic_quality` dimension directly. This is
    a tunable choice — alternatives include the weighted overall σ minus
    a constant, or a dedicated learned therapy-quality head.

    Args:
        judge_result: dict with keys "therapeutic_quality", optionally
            other dimensions and "safety_score".
        alpha_quality: scalar multiplier on therapeutic_quality. Set
            ≠1 to rebalance vs the consensus penalty.

    Returns:
        Scalar reward for the therapy-effectiveness channel.

    TODO when implementing:
      - Decide on the formulation: pure therapeutic_quality, or
        (overall σ + therapeutic_quality)/2, or learned head.
      - Consider trajectory-level rewards (e.g., did σ improve over
        the conversation?) — that's a function of multiple turns and
        belongs in rollout.py.
    """
    raise NotImplementedError


def r_joint(
    therapist_hidden: np.ndarray,
    monitor_hidden: np.ndarray,
    sigma: float,
    judge_result: dict,
    beta: float = 1.0,
    tau: float = 0.1,
    alpha_quality: float = 1.0,
) -> dict:
    """Proposal Eq. 3 — joint reward.

        R = r_therapy − β · c_consensus

    Returns a dict so the trainer can log all components separately:
        {"r_therapy": ..., "c_consensus": ..., "r_joint": ...}
    """
    raise NotImplementedError
