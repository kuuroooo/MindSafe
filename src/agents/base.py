import json
import re
from typing import Optional, List, Dict, Tuple

import numpy as np


def parse_json_response(text: str, default: dict) -> dict:
    if not text:
        return dict(default)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return dict(default)
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else dict(default)
    except json.JSONDecodeError:
        return dict(default)


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
