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
    name = "none"

    def transform_monitor_input(
        self,
        therapist_response: str,
        user_message: str,
        history,
    ) -> str:
        return therapist_response


class HookChain:
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
