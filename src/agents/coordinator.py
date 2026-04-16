from typing import List, Dict, Any, Optional

from .base import BaseAgent, format_history, parse_json_response


class CoordinatorAgent(BaseAgent):
    async def analyze(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        prompt = (
            f"Conversation so far:\n{format_history(chat_history)}\n\n"
            f"Latest user message:\n{user_message}\n\n"
            "Analyze this message and return a JSON object with keys:\n"
            '  "emotional_signals": list of short strings,\n'
            '  "risk_level": one of "low", "medium", "high",\n'
            '  "key_concerns": list of short strings,\n'
            '  "recommended_focus": single short string.\n'
            "Return only the JSON object."
        )
        raw = await self._generate(prompt, temperature=0.3, max_tokens=512)
        result = parse_json_response(raw, default={
            "emotional_signals": [],
            "risk_level": "low",
            "key_concerns": [],
            "recommended_focus": "",
        })
        result["raw"] = raw
        return result
