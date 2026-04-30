"""Inter-agent message hooks for red-teaming the oversight channel.

Two transform points, matching the two inter-agent surfaces an attack can
target:

  1. transform_monitor_input — what the *monitor* sees as the therapist's
     response (Za et al., Persuade Me If You Can, 2025).

  2. transform_coord_input  — what the *coordinator* sees just before the
     route() decision: the therapist response, the monitor's verdict
     payload, and the coordinator's own earlier analysis. This is the
     proposal's "communication attack" surface (He et al., Red-Teaming
     MAS via Communication Attacks, 2025) — a direct attack on the
     inter-agent message rather than indirect via monitor inference.

The user, the external judge, and the therapist's own next attempt all
see the *unaltered* response, so judge σ(s_t) stays ground truth for
what the user receives.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple


class AdversaryHook(Protocol):
    name: str

    def transform_monitor_input(
        self,
        therapist_response: str,
        user_message: str,
        history,
    ) -> str: ...

    def transform_coord_input(
        self,
        therapist_response: str,
        monitor_result: dict,
        analysis: dict,
        history,
    ) -> Tuple[str, dict, dict]: ...


# Backward-compat alias for prior callers that imported MonitorViewHook.
MonitorViewHook = AdversaryHook


class IdentityHook:
    """No-op hook. Default for the `clean` and `psi` ablation arms."""

    name = "none"

    def transform_monitor_input(
        self,
        therapist_response: str,
        user_message: str,
        history,
    ) -> str:
        return therapist_response

    def transform_coord_input(
        self,
        therapist_response: str,
        monitor_result: dict,
        analysis: dict,
        history,
    ) -> Tuple[str, dict, dict]:
        return therapist_response, monitor_result, analysis


class HookChain:
    """Compose multiple hooks left-to-right.

    Each transform point is run sequentially through the chain — useful
    for composing monitor + coord attacks, or two coord-side attacks.
    """

    def __init__(self, hooks: Optional[List[AdversaryHook]] = None):
        self.hooks = hooks or []
        self.name = "+".join(h.name for h in self.hooks) if self.hooks else "none"

    def transform_monitor_input(
        self,
        therapist_response: str,
        user_message: str,
        history,
    ) -> str:
        out = therapist_response
        for h in self.hooks:
            out = h.transform_monitor_input(out, user_message, history)
        return out

    def transform_coord_input(
        self,
        therapist_response: str,
        monitor_result: dict,
        analysis: dict,
        history,
    ) -> Tuple[str, dict, dict]:
        r, m, a = therapist_response, monitor_result, analysis
        for h in self.hooks:
            r, m, a = h.transform_coord_input(r, m, a, history)
        return r, m, a
