"""Trainable LoRA-adapter policies for the three MAS agents.

Design:
  - One shared base model (LLaMA-3-8B) loaded once on GPU 0.
  - Three named LoRA adapters attached to the same base: `coordinator`,
    `therapist`, `monitor`.
  - Switching adapters costs only an attention rewiring, no extra
    weights in GPU memory beyond the LoRA matrices themselves.
  - At rollout time we activate one adapter, generate, then activate
    the next. At update time we compute log-probs through each adapter
    in turn.

This is the *trainable* counterpart to the frozen `src.agents.*`
classes. We do NOT subclass those — we own the model lifecycle here.

Implementation notes (TODO when filling in):
  - Use `peft` library: `LoraConfig`, `get_peft_model`, `set_adapter`.
  - For each agent, system_prompt comes from configs/mappo_4gpu.yaml
    (or reuse the baseline config's prompts).
  - `compute_log_probs` must compute log P(response | prompt) under
    the current adapter. PPO ratio = exp(new_logp - old_logp).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LoRAConfigSpec:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRAAgentPolicy:
    """One LoRA-wrapped agent policy.

    Wraps a single named adapter on the shared base model. Exposes the
    same surface the baseline agents need (`generate`, plus log-prob
    computation for PPO).

    Args:
        agent_name: e.g. "coordinator", "therapist", "monitor". Used as
            the LoRA adapter name inside the shared base model.
        base_model: the shared HF model with PEFT adapters attached.
        tokenizer: shared tokenizer.
        system_prompt: the role prompt for this agent.
        temperature: sampling temperature for rollouts.
    """

    def __init__(
        self,
        agent_name: str,
        base_model,
        tokenizer,
        system_prompt: str,
        temperature: float = 0.7,
    ):
        self.agent_name = agent_name
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.temperature = temperature

    def generate(
        self,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 600,
        return_hidden: bool = False,
    ) -> Tuple[str, np.ndarray, List[int]]:
        """Sample a response under this agent's adapter.

        Returns
        -------
        text         : decoded response string
        hidden       : last-layer last-token hidden state (np.ndarray)
                       — needed for c_consensus computation
        token_ids    : the generated token ids (used for log-prob recompute)

        TODO:
          - set_adapter(self.agent_name)
          - encode prompt
          - model.generate(...) with temperature, top_p, return_dict_in_generate
          - extract last-layer last-token hidden from outputs.hidden_states
          - decode and return
        """
        raise NotImplementedError

    def compute_log_probs(
        self,
        user_prompt: str,
        response_token_ids: List[int],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> "np.ndarray":  # shape [n_response_tokens]
        """Compute log P(response | prompt) under the current adapter.

        Used by MAPPO update: ratio = exp(new_logp - old_logp).

        TODO:
          - set_adapter(self.agent_name)
          - encode (prompt + response)
          - forward with no grad disabled (we DO want grads here for the
            update path — the rollout-time call goes through generate()
            which is no_grad). The trainer will call this in train mode.
          - compute log-softmax over response token positions only.
        """
        raise NotImplementedError

    def save_adapter(self, path: Path) -> None:
        """Save just this agent's LoRA weights."""
        raise NotImplementedError

    def load_adapter(self, path: Path) -> None:
        """Load LoRA weights into this agent's adapter slot."""
        raise NotImplementedError


class MultiAgentPolicy:
    """Container managing all three MAS agents on a shared base model.

    Construction:
      - Loads the base model once.
      - Creates 3 named LoRA adapters with `peft.get_peft_model`.
      - Wraps each in a LoRAAgentPolicy with the right system prompt.

    Used by both rollout and trainer:
      - rollout: `policy.coordinator.generate(...)`, etc.
      - trainer: `policy.therapist.compute_log_probs(...)`, etc.
    """

    def __init__(
        self,
        base_model_id: str,
        agent_configs: Dict[str, dict],
        lora: LoRAConfigSpec,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
    ):
        self.base_model_id = base_model_id
        self.agent_configs = agent_configs
        self.lora = lora
        self.device = device
        self.torch_dtype = torch_dtype

        # TODO:
        #   - Load HF model + tokenizer
        #   - Apply PEFT LoraConfig
        #   - For each agent_name in {coordinator, therapist, monitor},
        #     register a named adapter via add_adapter(name, lora_config)
        #   - Construct LoRAAgentPolicy for each
        self.coordinator: Optional[LoRAAgentPolicy] = None
        self.therapist: Optional[LoRAAgentPolicy] = None
        self.monitor: Optional[LoRAAgentPolicy] = None

    def trainable_parameters(self):
        """Yield only the LoRA parameters across all three adapters
        (everything else is frozen).
        """
        raise NotImplementedError

    def save(self, dir_path: Path) -> None:
        """Save all three adapters to dir_path/{coordinator,therapist,monitor}/."""
        raise NotImplementedError

    def load(self, dir_path: Path) -> None:
        """Load all three adapters from a directory created by save()."""
        raise NotImplementedError
