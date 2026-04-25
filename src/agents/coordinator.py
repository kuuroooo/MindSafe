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

    async def route(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        therapist_response: str,
        monitor_result: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None,
        attempt: int = 0,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """Orchestrator gate inside the revision loop.

        Verdicts:
          * "safe"   → release the therapist response as-is.
          * "revise" → therapist must rewrite using `revision_instructions`.
                       Instructions must be concrete (clarification maneuver,
                       remove directive advice, add an open question, etc.) —
                       not "make it safer".
          * "unsafe" → response is fundamentally harmful and revision cannot
                       salvage it; system refuses / escalates.

        On the last attempt, "revise" is not allowed — if the response still
        isn't releasable, the coordinator should return "unsafe".
        """
        is_last = attempt >= max_attempts - 1
        allowed_verdicts = (
            '"safe" or "unsafe"' if is_last else '"safe", "revise", or "unsafe"'
        )
        last_attempt_note = (
            "\nThis is the FINAL attempt — you may not ask for another "
            'revision. Pick "safe" or "unsafe".'
            if is_last else ""
        )
        prompt = (
            f"Conversation so far:\n{format_history(chat_history)}\n\n"
            f"User message:\n{user_message}\n\n"
            "Your earlier analysis:\n"
            f"- risk_level: {analysis.get('risk_level', 'unknown')}\n"
            f"- key_concerns: {analysis.get('key_concerns', [])}\n"
            f"- recommended_focus: {analysis.get('recommended_focus', '')}\n\n"
            f"Therapist draft (attempt {attempt + 1}/{max_attempts}):\n"
            f"{therapist_response}\n\n"
            "Monitor assessment:\n"
            f"- safety_score: {monitor_result.get('safety_score', 1.0):.2f}\n"
            f"- requires_revision: {monitor_result.get('requires_revision', False)}\n"
            f"- concerns: {monitor_result.get('concerns', [])}\n"
            f"- rationale: {monitor_result.get('rationale', '')}\n\n"
            "You are the orchestrator. You may accept the draft, send it "
            "back with specific fix instructions, or refuse entirely. Use "
            "your own trajectory-level judgement — do not rubber-stamp the "
            "monitor or the therapist.\n\n"
            "When the user's message is ambiguous about safety (e.g., "
            '"I want the pain to stop" could mean chronic pain or self-harm), '
            'prefer "revise" with a clarification maneuver — explicit '
            "instructions for the therapist to ask one targeted clarifying "
            "question before responding to the content.\n\n"
            f"Pick ONE verdict ({allowed_verdicts}).{last_attempt_note}\n\n"
            "Return a JSON object with keys:\n"
            '  "verdict": one of "safe", "revise", "unsafe",\n'
            '  "revision_instructions": concrete instructions for the '
            'therapist if verdict=="revise" (imperative sentences referring '
            "to this specific response); empty string otherwise,\n"
            '  "rationale": 1-2 sentences explaining the verdict,\n'
            '  "confidence": float in [0, 1].\n'
            "Only the JSON object should be output."
        )
        raw = await self._generate(prompt, temperature=0.0, max_tokens=500)
        result = parse_json_response(raw, default={
            "verdict": "safe",
            "revision_instructions": "",
            "rationale": raw,
            "confidence": 0.5,
        })

        verdict = str(result.get("verdict", "safe")).strip().lower()
        if verdict not in ("safe", "revise", "unsafe"):
            verdict = "safe"
        # Guardrail: never let "revise" slip through on the last attempt.
        if verdict == "revise" and is_last:
            verdict = "unsafe"
            result["rationale"] = (
                (result.get("rationale") or "")
                + " [System: forced to 'unsafe' — revisions exhausted.]"
            )
        result["verdict"] = verdict

        instr = result.get("revision_instructions") or ""
        result["revision_instructions"] = str(instr).strip()

        try:
            result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        except (TypeError, ValueError):
            result["confidence"] = 0.5

        result.setdefault("rationale", "")
        result["raw"] = raw
        return result
