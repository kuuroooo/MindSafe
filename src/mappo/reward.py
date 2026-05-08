"""Reward functions for MAPPO training (proposal §7.2 Eqs 3-4).

These are pure functions operating on a single turn's data. Imported
into `rollout.py` to label every step of every trajectory.

Sign conventions:
  * `c_consensus` is a *cost* (high = bad).
  * `r_therapy`   is a *reward* (high = good).
  * `r_joint`     is `r_therapy − β·c_consensus`. PPO maximizes this.

The `c_consensus` formula here is intentionally identical to the
baseline analysis in `scripts/consensus_penalty.py` — same cosine
distance, same kernel, same temperature default. So training-time
penalty and post-hoc baseline analysis are on the same scale.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


# -----------------------------------------------------------------------------
# Eq. 4 — unsafe-consensus cost
# -----------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance ∈ [0, 2]. Matches `ConsensusMetrics.latent_cosine_distance`."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def c_consensus_from_distance(d: float, sigma: float, tau: float = 0.1) -> float:
    """Eq. 4 with the cosine distance precomputed.

    Used by `consensus_penalty.py` and by post-hoc analysis on the
    baseline JSONLs (which store `latent_distance` already, not the
    raw hidden vectors).
    """
    return math.exp(-d / tau) * (1.0 - float(sigma))


def c_consensus(
    therapist_hidden: np.ndarray,
    monitor_hidden: np.ndarray,
    sigma: float,
    tau: float = 0.1,
) -> float:
    """Eq. 4 — unsafe-consensus cost.

        c = exp(-d(h_therapist, h_monitor) / τ) · (1 - σ)

    High when (a) the agents' last-layer last-token hidden states are
    in latent agreement (small d) AND (b) the judge rates the response
    unsafe (small σ). Bounded in [0, 1]: the exponential is in (0, 1]
    and (1−σ) is in [0, 1].
    """
    d = cosine_distance(therapist_hidden, monitor_hidden)
    return c_consensus_from_distance(d, sigma, tau)


# -----------------------------------------------------------------------------
# Therapeutic-utility reward (option A: judge's therapeutic_quality dim)
# -----------------------------------------------------------------------------

def r_therapy(judge_result: Dict, alpha_quality: float = 1.0) -> float:
    """Therapeutic utility from the judge's `therapeutic_quality` dimension.

    Option A from the design discussion: keeps therapy and safety as
    orthogonal channels so β has a clean interpretation. Falls back to
    a neutral 0.5 if the judge didn't emit the dimension.

    Args:
        judge_result: dict from `ExternalJudgeAgent.evaluate(...)`.
        alpha_quality: scalar multiplier. Default 1.0 keeps
            r_therapy in [0, 1], same scale as c_consensus.
    """
    q = judge_result.get("therapeutic_quality")
    if q is None:
        q = 0.5
    try:
        q = float(q)
    except (TypeError, ValueError):
        q = 0.5
    q = max(0.0, min(1.0, q))
    return alpha_quality * q


# -----------------------------------------------------------------------------
# Eq. 3 — joint reward
# -----------------------------------------------------------------------------

def r_joint(
    therapist_hidden: np.ndarray,
    monitor_hidden: np.ndarray,
    sigma: float,
    judge_result: Dict,
    beta: float = 1.0,
    tau: float = 0.1,
    alpha_quality: float = 1.0,
) -> Dict[str, float]:
    """Eq. 3 — joint reward used as PPO target.

        R = r_therapy − β · c_consensus

    Returns a dict so the trainer can log each component independently.
    PPO uses the `r_joint` value; logging dashboards show r_therapy and
    c_consensus separately.
    """
    rt = r_therapy(judge_result, alpha_quality=alpha_quality)
    cc = c_consensus(therapist_hidden, monitor_hidden, sigma, tau=tau)
    return {
        "r_therapy": rt,
        "c_consensus": cc,
        "r_joint": rt - beta * cc,
    }


def r_joint_from_distance(
    distance: float,
    sigma: float,
    judge_result: Dict,
    beta: float = 1.0,
    tau: float = 0.1,
    alpha_quality: float = 1.0,
) -> Dict[str, float]:
    """Convenience: same as `r_joint` but takes the precomputed distance.

    Useful for retroactive analysis on the baseline JSONLs (which store
    `latent_distance` rather than raw hidden vectors).
    """
    rt = r_therapy(judge_result, alpha_quality=alpha_quality)
    cc = c_consensus_from_distance(distance, sigma, tau=tau)
    return {
        "r_therapy": rt,
        "c_consensus": cc,
        "r_joint": rt - beta * cc,
    }
