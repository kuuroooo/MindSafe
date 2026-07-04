from __future__ import annotations

import math
from typing import Dict

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        # undefined direction: treat as orthogonal (distance 1.0)
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


# eq. 4: exp(-d/tau)*(1-sigma). kept identical to scripts/consensus_penalty.py so training and baseline share one scale
def c_consensus_from_distance(d: float, sigma: float, tau: float = 0.1) -> float:
    return math.exp(-d / tau) * (1.0 - float(sigma))


def c_consensus(
    therapist_hidden: np.ndarray,
    monitor_hidden: np.ndarray,
    sigma: float,
    tau: float = 0.1,
) -> float:
    d = cosine_distance(therapist_hidden, monitor_hidden)
    return c_consensus_from_distance(d, sigma, tau)


def r_therapy(judge_result: Dict, alpha_quality: float = 1.0) -> float:
    q = judge_result.get("therapeutic_quality")
    if q is None:
        q = 0.5  # neutral fallback when the judge omits the dimension
    try:
        q = float(q)
    except (TypeError, ValueError):
        q = 0.5
    q = max(0.0, min(1.0, q))
    return alpha_quality * q


def _select_penalty(c_cons: float, sigma: float, penalty_form: str) -> float:
    if penalty_form == "c_consensus":
        return c_cons
    if penalty_form == "unsafety_only":
        return 1.0 - float(sigma)
    raise ValueError(f"unknown penalty_form: {penalty_form!r}")


def r_joint(
    therapist_hidden: np.ndarray,
    monitor_hidden: np.ndarray,
    sigma: float,
    judge_result: Dict,
    beta: float = 1.0,
    tau: float = 0.1,
    alpha_quality: float = 1.0,
    penalty_form: str = "c_consensus",
) -> Dict[str, float]:
    rt  = r_therapy(judge_result, alpha_quality=alpha_quality)
    cc  = c_consensus(therapist_hidden, monitor_hidden, sigma, tau=tau)
    pen = _select_penalty(cc, sigma, penalty_form)
    return {
        "r_therapy": rt,
        "c_consensus": cc,
        "penalty": pen,
        "penalty_form": penalty_form,
        "r_joint": rt - beta * pen,
    }


def r_joint_from_distance(
    distance: float,
    sigma: float,
    judge_result: Dict,
    beta: float = 1.0,
    tau: float = 0.1,
    alpha_quality: float = 1.0,
    penalty_form: str = "c_consensus",
) -> Dict[str, float]:
    rt  = r_therapy(judge_result, alpha_quality=alpha_quality)
    cc  = c_consensus_from_distance(distance, sigma, tau=tau)
    pen = _select_penalty(cc, sigma, penalty_form)
    return {
        "r_therapy": rt,
        "c_consensus": cc,
        "penalty": pen,
        "penalty_form": penalty_form,
        "r_joint": rt - beta * pen,
    }
