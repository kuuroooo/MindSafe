"""Centralized value function V_φ(s_t) for MAPPO.

Per proposal §7.2: a centralized critic V_φ that takes the global
state (concatenated context of all agents) and outputs a scalar.
Used by GAE to compute the advantage Â_t shared across all agents.

Implementation: option A from the design discussion — frozen base
model + tiny MLP head on mean-pooled last-layer hidden states.

Trainable parameters: just the head. Base model stays frozen and is
shared with the policies' adapters (those don't activate during the
critic forward — we explicitly disable adapters before pooling).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn

from .policy import disable_all_adapters_ctx


def encode_global_state(
    chat_history: List[Dict[str, str]],
    user_message: str,
    coord_analysis: Optional[dict] = None,
    max_history_turns: int = 8,
) -> str:
    """Concatenate the visible-to-the-triad context into one string.

    The "global state" per proposal §7.2 is everything visible to the
    agents at decision time. We include the recent chat history, the
    latest user message, and the coordinator's analysis (if available).

    Truncated history avoids unbounded context growth — recent turns
    carry the most state-relevant information for value estimation.
    """
    out = ["[Global state]"]
    if chat_history:
        out.append("Conversation:")
        for t in chat_history[-max_history_turns:]:
            role = t.get("role", "?")
            content = t.get("content", "")
            out.append(f"  {role}: {content}")
    out.append(f"User: {user_message}")
    if coord_analysis:
        out.append(
            "Coordinator analysis: "
            f"risk_level={coord_analysis.get('risk_level','?')}; "
            f"key_concerns={coord_analysis.get('key_concerns', [])}; "
            f"recommended_focus={coord_analysis.get('recommended_focus','')}"
        )
    return "\n".join(out)


class ValueHead(nn.Module):
    """2-layer MLP that maps a pooled hidden vector to a scalar V(s).

    Init: small. The output should start near zero so early-training
    advantages are dominated by the actual reward signal, not a noisy
    critic prior.
    """

    def __init__(self, hidden_dim: int, head_hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, head_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(head_hidden, 1)
        # Small init on output layer — keeps V(s) ≈ 0 at start
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        # pooled_hidden: [batch, hidden_dim] → [batch]
        return self.fc2(self.act(self.fc1(pooled_hidden))).squeeze(-1)


class CentralizedValueNet:
    """V_φ(s_t) — frozen base + trainable MLP head.

    Forward pass:
      1. Encode the global state to a string.
      2. Tokenize.
      3. Forward through base model with NO adapter active and
         output_hidden_states=True. The base is frozen, so we run
         under torch.no_grad() to save memory.
      4. Mean-pool last-layer hidden states across the (non-padded)
         sequence.
      5. The trainable head maps that to V(s).

    Note: the base-model forward is no_grad, but the head's forward
    needs grad enabled (for backprop on value loss). We detach the
    pooled hidden states before the head.
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        hidden_dim: int = 4096,
        head_hidden: int = 512,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.head = ValueHead(hidden_dim, head_hidden).to(device).to(torch_dtype)
        self._max_len = 4096  # truncation cap for the global-state string

    def _pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool the base model's last-layer hidden states over real tokens."""
        with torch.no_grad():
            with disable_all_adapters_ctx(self.base_model):
                out = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
        last = out.hidden_states[-1]                         # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(last.dtype)   # [B, T, 1]
        summed = (last * mask).sum(dim=1)                    # [B, H]
        denom = mask.sum(dim=1).clamp(min=1.0)               # [B, 1]
        return (summed / denom).detach()

    def __call__(self, global_state_text: str) -> float:
        """V(s) for a single string. Inference path — no grad needed."""
        with torch.no_grad():
            return float(self.batched([global_state_text])[0].detach())

    def batched(self, global_states: List[str]) -> torch.Tensor:
        """V(s) for a batch — used in the trainer's value loss.

        Returns a tensor [B] with grad enabled on the head parameters.
        """
        toks = self.tokenizer(
            global_states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_len,
        ).to(self.device)
        pooled = self._pool(toks["input_ids"], toks["attention_mask"])
        # head is trainable; pooled is detached so grads only flow into head
        return self.head(pooled.to(self.head.fc1.weight.dtype))

    def trainable_parameters(self):
        return list(self.head.parameters())

    # ---- checkpoint -------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path); dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), dir_path / "value_head.pt")

    def load(self, dir_path: Path) -> None:
        state = torch.load(Path(dir_path) / "value_head.pt", map_location=self.device)
        self.head.load_state_dict(state)
