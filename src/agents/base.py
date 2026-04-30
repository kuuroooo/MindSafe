import json
import re
from typing import Optional, List, Dict, Tuple

import numpy as np


def parse_json_response(text: str, default: dict) -> dict:
    """Extract a JSON object from a (possibly noisy) LLM response.

    Handles three failure modes the previous greedy-regex parser missed:

    1. Multiple top-level objects (e.g., the LLM emits `{...}\n\n{...}`):
       try each balanced `{...}` block in order, return the first that
       parses as a dict.
    2. Code-fenced JSON (```json ... ```): extract from the fence first.
    3. Invalid JSON the LLM emits anyway (most often: unescaped double
       quotes inside a string field). When no full parse succeeds, fall
       back to per-key regex salvage so we still recover the fields we
       care about — picking the LAST occurrence of each key, which in
       practice is the LLM's "real" answer (it tends to emit the wrong
       JSON first, then a corrected version).

    The previous version silently returned `default` for these cases,
    which masked real coordinator/monitor verdicts (the parser-bug case
    we observed at σ=0.60: coord wanted to refuse with verdict="revise"
    but unescaped quotes in its rationale broke the outer parse, so the
    system saw default verdict="safe" and released the response).
    """
    if not text:
        return dict(default)

    # 1. Code-fenced JSON
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 2. Each balanced {...} block, in order
    for obj_text in _balanced_objects(text):
        try:
            obj = json.loads(obj_text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # 3. Per-key regex salvage — last match wins
    return _salvage_keys(text, default)


def _balanced_objects(text: str):
    """Yield each top-level balanced `{...}` substring in `text`.

    Tracks string state (with backslash escapes) so braces inside strings
    don't perturb depth counting.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    yield text[start:i + 1]
                    start = -1


_NUM_OR_BOOL = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)"


def _salvage_keys(text: str, default: dict) -> dict:
    """Last-resort regex extraction when full JSON parse fails.

    For each expected key in `default`, scan `text` for `"key": value`
    occurrences and use the LAST one. Strings are matched with the
    standard escape-aware pattern; numbers/bools/null parsed via json.loads
    on the captured token. List/dict values aren't salvaged — they keep
    the default.
    """
    result = dict(default)
    for key in default:
        key_pat = re.escape(key)

        # String-typed value
        ms = list(re.finditer(
            rf'"{key_pat}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            text,
        ))
        if ms:
            try:
                result[key] = json.loads(f'"{ms[-1].group(1)}"')
            except json.JSONDecodeError:
                result[key] = ms[-1].group(1)
            continue

        # Numeric / bool / null value
        ms = list(re.finditer(rf'"{key_pat}"\s*:\s*{_NUM_OR_BOOL}', text))
        if ms:
            try:
                result[key] = json.loads(ms[-1].group(1))
            except json.JSONDecodeError:
                pass

    return result


def format_history(history: Optional[List[Dict[str, str]]], max_turns: int = 6) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(
        f"{t.get('role', '?')}: {t.get('content', '')}"
        for t in history[-max_turns:]
    )


class BaseAgent:
    def __init__(self, config: dict, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.system_prompt = config.get("system_prompt", "")
        self.temperature = config.get("temperature", 0.7)

    async def _generate(
        self,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> str:
        return await self.llm_client.generate_async(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )

    async def _generate_with_hidden(
        self,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> Tuple[str, np.ndarray]:
        """Generate text and return (text, latent hidden-state vector).

        Requires the underlying client to expose `generate_with_hidden_async`.
        Used by Therapist and Monitor for latent-space consensus analysis.
        """
        fn = getattr(self.llm_client, "generate_with_hidden_async", None)
        if fn is None:
            raise RuntimeError(
                f"{type(self.llm_client).__name__} does not expose hidden states; "
                "Therapist and Monitor must run on HFClient."
            )
        return await fn(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )
