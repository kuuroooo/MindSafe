from .hooks import HookChain, IdentityHook, MonitorViewHook
from .monitor_persuasion import MonitorPersuasionAttack


def build_hook(spec) -> MonitorViewHook:
    """Factory mapping a config entry (`hook:` field) to a hook instance.

    Accepts:
      - None / "none" / missing: IdentityHook
      - dict with `type: monitor_persuasion, mode: ...`
      - list of the above (composed via HookChain)
    """
    if spec is None or spec == "none":
        return IdentityHook()
    if isinstance(spec, list):
        return HookChain([build_hook(s) for s in spec])
    if not isinstance(spec, dict):
        raise ValueError(f"Unrecognized hook spec: {spec!r}")
    htype = spec.get("type")
    if htype == "monitor_persuasion":
        return MonitorPersuasionAttack(mode=spec.get("mode", "justification_only"))
    raise ValueError(f"Unknown hook type: {htype!r}")


__all__ = [
    "HookChain",
    "IdentityHook",
    "MonitorViewHook",
    "MonitorPersuasionAttack",
    "build_hook",
]
