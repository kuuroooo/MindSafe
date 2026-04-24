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

    async def finalize(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        therapist_response: str,
        monitor_result: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Final orchestration gate after Therapist + Monitor.

        The coordinator synthesizes its own prior analysis, the therapist's
        response, and the monitor's assessment into a single release decision.
        Returns {final_label: "safe"|"unsafe", confidence, rationale}.
        """
        prompt = (
            f"Conversation so far:\n{format_history(chat_history)}\n\n"
            f"User message:\n{user_message}\n\n"
            "Your earlier analysis:\n"
            f"- risk_level: {analysis.get('risk_level', 'unknown')}\n"
            f"- key_concerns: {analysis.get('key_concerns', [])}\n"
            f"- recommended_focus: {analysis.get('recommended_focus', '')}\n\n"
            f"Therapist response:\n{therapist_response}\n\n"
            "Monitor assessment:\n"
            f"- safety_score: {monitor_result.get('safety_score', 1.0):.2f}\n"
            f"- requires_revision: {monitor_result.get('requires_revision', False)}\n"
            f"- concerns: {monitor_result.get('concerns', [])}\n"
            f"- rationale: {monitor_result.get('rationale', '')}\n\n"
            "You are the orchestrator. Decide whether the therapist response "
            "should be released to the user. Consider the conversation trajectory, "
            "your own risk analysis, and the monitor's concerns — do not simply "
            "defer to the monitor.\n\n"
            "Return a JSON object with keys:\n"
            '  "final_label": "safe" or "unsafe",\n'
            '  "confidence": float in [0, 1],\n'
            '  "rationale": short paragraph.\n'
            "Only the JSON object should be output."
        )
        raw = await self._generate(prompt, temperature=0.0, max_tokens=400)
        result = parse_json_response(raw, default={
            "final_label": "safe",
            "confidence": 0.5,
            "rationale": raw,
        })
        label = str(result.get("final_label", "safe")).strip().lower()
        result["final_label"] = "unsafe" if label == "unsafe" else "safe"
        try:
            c = float(result.get("confidence", 0.5))
        except (TypeError, ValueError):
            c = 0.5
        result["confidence"] = max(0.0, min(1.0, c))
        result.setdefault("rationale", "")
        result["raw"] = raw
        return result
