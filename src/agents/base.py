import json
import re
from typing import Optional, List, Dict, Tuple

import numpy as np


# pull a json object out of noisy llm output: try a code fence, then any balanced {...}, then per-key regex salvage
def parse_json_response(text: str, default: dict) -> dict:
    if not text:
        return dict(default)

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    for obj_text in _balanced_objects(text):
        try:
            obj = json.loads(obj_text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return _salvage_keys(text, default)


def _balanced_objects(text: str):
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


# last-resort extraction when json.loads fails; ms[-1] wins because models often restate a key
def _salvage_keys(text: str, default: dict) -> dict:
    result = dict(default)
    for key in default:
        key_pat = re.escape(key)

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
