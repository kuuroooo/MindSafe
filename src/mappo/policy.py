"""Trainable LoRA-adapter policies for the three MAS agents.

One shared base model. Three named LoRA adapters: `coordinator`,
`therapist`, `monitor`. Switching adapters is an attention rewiring,
not a weight reload. The base model stays frozen — only LoRA matrices
are trained.

This module owns the model lifecycle. The frozen `src.agents.*`
classes are unaffected.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class LoRAConfigSpec:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


# -----------------------------------------------------------------------------
# Adapter context helpers
# -----------------------------------------------------------------------------

@contextlib.contextmanager
def active_adapter_ctx(model, adapter_name: str):
    """Temporarily activate one named LoRA adapter."""
    prev = getattr(model, "active_adapter", None)
    model.set_adapter(adapter_name)
    try:
        yield
    finally:
        if prev is not None and prev != adapter_name:
            try:
                model.set_adapter(prev)
            except Exception:
                pass


@contextlib.contextmanager
def disable_all_adapters_ctx(model):
    """Run a forward pass against the frozen base only.

    Used by the value net (so its features come from the un-adapted
    base) and by the patient simulator (so user messages aren't biased
    by whatever adapter happens to be active).
    """
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


# -----------------------------------------------------------------------------
# Single-agent policy
# -----------------------------------------------------------------------------

class LoRAAgentPolicy:
    """Wraps one named adapter on the shared base model.

    Exposes the two operations PPO needs:
      - `generate`: sample a response (rollout-time, no_grad).
      - `compute_log_probs`: log P(response | prompt) under the
        *current* adapter weights (update-time, with grad).
    """

    def __init__(
        self,
        agent_name: str,
        base_model: PeftModel,
        tokenizer,
        system_prompt: str,
        device: str = "cuda:0",
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.agent_name = agent_name
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.device = device
        self.temperature = temperature
        self.top_p = top_p

    # ---- prompt construction ----------------------------------------------

    def _build_prompt(
        self,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ---- rollout-time: generate -------------------------------------------

    def generate(
        self,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 600,
        return_hidden: bool = False,
    ) -> Dict:
        """Sample a response under this agent's adapter.

        Returns a dict:
          {
            "text":        str,        # decoded generation
            "prompt":      str,        # the rendered prompt fed to the model
            "prompt_ids":  Tensor,     # input_ids for the prompt
            "response_ids": Tensor,    # new token ids (no prompt)
            "log_probs":   Tensor,     # log P(token) at sample time, [n_resp]
            "hidden":      np.ndarray, # last-layer last-token hidden (if return_hidden)
          }
        """
        prompt = self._build_prompt(user_prompt, chat_history)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,           # so we can compute sample-time log-probs
        )
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with active_adapter_ctx(self.base_model, self.agent_name):
            with torch.no_grad():
                out = self.base_model.generate(**inputs, **gen_kwargs)

        seqs = out.sequences[0]
        response_ids = seqs[prompt_len:]
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # Sample-time per-token log-probs (used as π_θ_old in PPO ratio)
        log_probs = self._collect_sample_log_probs(out.scores, response_ids)

        result = {
            "text": text,
            "prompt": prompt,
            "prompt_ids": inputs["input_ids"][0].detach().cpu(),
            "response_ids": response_ids.detach().cpu(),
            "log_probs": log_probs.detach().cpu(),
        }

        if return_hidden:
            # Single forward pass on full (prompt+response) under the agent's
            # adapter to extract last-layer last-token hidden state.
            full_ids = seqs.unsqueeze(0)
            with active_adapter_ctx(self.base_model, self.agent_name):
                with torch.no_grad():
                    fwd = self.base_model(
                        input_ids=full_ids,
                        output_hidden_states=True,
                        use_cache=False,
                    )
            hidden = fwd.hidden_states[-1][0, -1, :].detach().to(torch.float32).cpu().numpy()
            result["hidden"] = hidden

        return result

    @staticmethod
    def _collect_sample_log_probs(scores, response_ids: torch.Tensor) -> torch.Tensor:
        """Per-token log P at sample time.

        `scores` is a tuple of length n_new_tokens; each entry is the
        logits for that step over vocab. We sample-time log-prob each
        actually-emitted token.
        """
        lps = []
        for step, logits in enumerate(scores):
            log_softmax = torch.log_softmax(logits[0], dim=-1)
            tok = response_ids[step]
            lps.append(log_softmax[tok])
        if not lps:
            return torch.zeros(0)
        return torch.stack(lps)

    # ---- update-time: compute log-probs (differentiable) ------------------

    def compute_log_probs(
        self,
        prompt_ids: torch.Tensor,        # [n_prompt]
        response_ids: torch.Tensor,      # [n_resp]
    ) -> torch.Tensor:
        """log P(response | prompt) under the CURRENT adapter weights.

        Differentiable — used inside the PPO update to compute
        ratio = exp(new_log_probs - old_log_probs) and the clipped
        surrogate.

        Returns: per-token log-probs, shape [n_resp].
        """
        full = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0).to(self.device)
        # The model predicts token t+1 from positions ≤ t. So log P(response[i])
        # comes from logits at position (n_prompt - 1 + i).
        with active_adapter_ctx(self.base_model, self.agent_name):
            out = self.base_model(input_ids=full, use_cache=False)
        logits = out.logits[0]                           # [T, V]
        n_prompt = prompt_ids.shape[0]
        # logits at position k predict token k+1; we want the predictions
        # for response_ids[0..n_resp-1], i.e. logits[n_prompt-1 : n_prompt-1 + n_resp]
        target_logits = logits[n_prompt - 1 : n_prompt - 1 + response_ids.shape[0]]  # [n_resp, V]
        log_softmax = torch.log_softmax(target_logits, dim=-1)
        targets = response_ids.to(self.device)
        return log_softmax.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [n_resp]


# -----------------------------------------------------------------------------
# Multi-agent container
# -----------------------------------------------------------------------------

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class MultiAgentPolicy:
    """Loads the shared base model, attaches three named LoRA adapters,
    and exposes one `LoRAAgentPolicy` per agent.

    Construction does the heavy I/O (model load, adapter init). Use it
    as a context for the trainer:

        policy = MultiAgentPolicy(model_id, agent_configs, lora=...)
        policy.coordinator.generate(...)
        policy.therapist.generate(..., return_hidden=True)
        ...
    """

    AGENT_NAMES = ("coordinator", "therapist", "monitor")

    def __init__(
        self,
        base_model_id: str,
        agent_configs: Dict[str, dict],
        lora: LoRAConfigSpec,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 600,
    ):
        self.base_model_id = base_model_id
        self.agent_configs = agent_configs
        self.lora_spec = lora
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.torch_dtype = _DTYPES.get(torch_dtype, torch.bfloat16)
        self.max_new_tokens = max_new_tokens

        # ---- tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- base model ----
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=self.torch_dtype,
            device_map={"": device},
        )

        # ---- attach the FIRST adapter via get_peft_model, then add the others ----
        first_name = self.AGENT_NAMES[0]
        lora_cfg = self._make_lora_config()
        self.base_model: PeftModel = get_peft_model(base, lora_cfg, adapter_name=first_name)
        for name in self.AGENT_NAMES[1:]:
            self.base_model.add_adapter(name, self._make_lora_config())

        # Memory: gradient checkpointing trades ~30% compute for ~3× activation
        # memory savings. Required for Llama-3-8B + LoRA + PPO on a single
        # 40GB A100. PEFT requires `enable_input_require_grads` so gradient
        # propagates through the embedding layer.
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()

        # ---- per-agent policy wrappers ----
        self.coordinator = self._make_agent("coordinator")
        self.therapist = self._make_agent("therapist")
        self.monitor = self._make_agent("monitor")

    def _make_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.lora_spec.rank,
            lora_alpha=self.lora_spec.alpha,
            lora_dropout=self.lora_spec.dropout,
            target_modules=list(self.lora_spec.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _make_agent(self, name: str) -> LoRAAgentPolicy:
        cfg = self.agent_configs.get(name, {}) or {}
        return LoRAAgentPolicy(
            agent_name=name,
            base_model=self.base_model,
            tokenizer=self.tokenizer,
            system_prompt=cfg.get("system_prompt", ""),
            device=self.device,
            temperature=float(cfg.get("temperature", 0.7)),
        )

    # ---- training utilities -----------------------------------------------

    def trainable_parameters(self):
        """Yield all LoRA matrices across the three adapters."""
        for p in self.base_model.parameters():
            if p.requires_grad:
                yield p

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ---- checkpoint -------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        """Save all three adapters into dir_path/{agent_name}/.

        Single save_pretrained call covering every adapter — PEFT writes
        one subdirectory per adapter name. Per-adapter calls with
        selected_adapters=[name] double-nest as dir/name/name/, which
        breaks load_adapter (looks one level shallower).
        """
        dir_path = Path(dir_path).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(
            str(dir_path),
            selected_adapters=list(self.AGENT_NAMES),
        )

    def load(self, dir_path: Path) -> None:
        """Load adapter weights from dir_path/{agent_name}/.

        Assumes adapter slots are already created (via __init__). Only
        replaces weights, not adapter structure. Absolute path forces
        PEFT to treat this as a local dir rather than HF repo id.
        """
        dir_path = Path(dir_path).resolve()
        for name in self.AGENT_NAMES:
            self.base_model.load_adapter(str(dir_path / name), adapter_name=name)
