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
    def __init__(self, hidden_dim: int, head_hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, head_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(head_hidden, 1)
        # small init keeps V(s) ~ 0 at the start of training
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(pooled_hidden))).squeeze(-1)


class CentralizedValueNet:
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
        self._max_len = 4096

    def _pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with disable_all_adapters_ctx(self.base_model):
                out = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
        last = out.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(last.dtype)
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        # detach: grads train only the value head, never the frozen base
        return (summed / denom).detach()

    def __call__(self, global_state_text: str) -> float:
        with torch.no_grad():
            return float(self.batched([global_state_text])[0].detach())

    def batched(self, global_states: List[str]) -> torch.Tensor:
        toks = self.tokenizer(
            global_states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_len,
        ).to(self.device)
        pooled = self._pool(toks["input_ids"], toks["attention_mask"])
        return self.head(pooled.to(self.head.fc1.weight.dtype))

    def trainable_parameters(self):
        return list(self.head.parameters())


    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path); dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), dir_path / "value_head.pt")

    def load(self, dir_path: Path) -> None:
        state = torch.load(Path(dir_path) / "value_head.pt", map_location=self.device)
        self.head.load_state_dict(state)
