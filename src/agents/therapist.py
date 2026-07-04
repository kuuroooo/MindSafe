from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from .base import BaseAgent, format_history


class TherapistAgent(BaseAgent):
    def __init__(self, config: dict, llm_client):
        super().__init__(config, llm_client)
        self.constitution: List[str] = config.get("constitution", [])
        self.max_revisions: int = config.get("max_revisions", 2)
        if self.constitution:
            principles = "\n".join(f"- {p}" for p in self.constitution)
            self.system_prompt = (
                f"{self.system_prompt}\n\nGuiding principles:\n{principles}"
            )

    async def respond(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None,
        revision_feedback: Optional[str] = None,
    ) -> Tuple[str, np.ndarray]:
        parts = [f"Prior conversation:\n{format_history(chat_history)}"]

        if analysis:
            parts.append(
                "Coordinator analysis:\n"
                f"- risk_level: {analysis.get('risk_level', 'unknown')}\n"
                f"- key_concerns: {analysis.get('key_concerns', [])}\n"
                f"- recommended_focus: {analysis.get('recommended_focus', '')}"
            )

        parts.append(f"User: {user_message}")

        if revision_feedback:
            parts.append(
                "Your previous response was flagged with these concerns:\n"
                f"{revision_feedback}\n"
                "Rewrite the response addressing them."
            )

        parts.append("Respond to the user. Reply with only the response text.")
        prompt = "\n\n".join(parts)
        return await self._generate_with_hidden(prompt, max_tokens=600)
