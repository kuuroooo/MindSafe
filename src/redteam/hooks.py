"""Inter-agent message hooks for red-teaming the oversight channel.

Attacks warp the *monitor's view* of the therapist's response. The user,
coordinator.finalize, and external judge all see the unaltered response,
so the judge's safety score stays ground-truth of what the user receives
(matching Za et al., Persuade Me If You Can, 2025).
"""

from __future__ import annotations

from typing import List, Optional, Protocol


class MonitorViewHook(Protocol):
    name: str

    def transform_monitor_input(
        self,
        therapist_response: str,
        user_message: str,
        history,
    ) -> str: ...


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


class HookChain:
    """Compose multiple monitor-view hooks left-to-right.

    Baseline uses at most one active hook, but this keeps the interface
    future-proof for the identity-bias + persuasion combination arms
    planned for later phases.
    """

    def __init__(self, hooks: Optional[List[MonitorViewHook]] = None):
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
