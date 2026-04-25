"""HuggingFace transformers client pinned to a single GPU."""

import asyncio
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class HFModelConfig(BaseModel):
    model_id: str
    device: Optional[str] = None
    device_map: Optional[str] = None
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    trust_remote_code: bool = False
    max_new_tokens: int = 1024


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HFClient:
    def __init__(self, config: HFModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        print(f"[HF] Loading {self.config.model_id}")
        if self.config.device:
            print(f"[HF]   device={self.config.device}")

        quant_cfg = None
        if self.config.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Newer transformers prefers `dtype` (`torch_dtype` is deprecated).
        kwargs = {
            "dtype": _DTYPES.get(self.config.torch_dtype, torch.bfloat16),
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.device:
            kwargs["device_map"] = {"": self.config.device}
        elif self.config.device_map:
            kwargs["device_map"] = self.config.device_map
        if quant_cfg is not None:
            kwargs["quantization_config"] = quant_cfg
        if self.config.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, **kwargs
        )

        if hasattr(self.model, "hf_device_map"):
            placements = set(str(v) for v in self.model.hf_device_map.values())
            print(f"[HF]   placements: {placements}")
        else:
            print(f"[HF]   placement: {next(self.model.parameters()).device}")

    def _build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        max_tokens = max_tokens or self.config.max_new_tokens
        prompt = self._build_prompt(system_prompt, user_prompt, chat_history)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def generate_async(self, *args, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(*args, **kwargs))

    def generate_with_hidden(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, np.ndarray]:
        """Generate and return (text, last-layer last-token hidden state).

        The hidden vector is the last-layer representation at the final token
        of the full (prompt + generated) sequence — the model's internal
        state just after producing its response.
        """
        max_tokens = max_tokens or self.config.max_new_tokens
        prompt = self._build_prompt(system_prompt, user_prompt, chat_history)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        sequences = outputs.sequences
        new_tokens = sequences[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # hidden_states: tuple[n_new_tokens], each a tuple[n_layers+1] of
        # tensors [batch, seq, hidden_dim]. Take the last-generated step,
        # last layer, last token position.
        last_step = outputs.hidden_states[-1]
        last_layer = last_step[-1]
        hidden = last_layer[0, -1, :].detach().to(torch.float32).cpu().numpy()
        return text, hidden

    async def generate_with_hidden_async(
        self, *args, **kwargs
    ) -> Tuple[str, np.ndarray]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate_with_hidden(*args, **kwargs)
        )
