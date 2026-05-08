"""Centralized value function V_φ(s_t) for MAPPO.

Per proposal §7.2:
  > We estimate a centralized Value Function V_φ(s_t) that takes the
  > global state s_t (concatenated context of all agents) to compute
  > the advantage Â_t.

Design choice: read out from the *base model's* hidden states rather
than train a separate transformer from scratch. The base model already
encodes the conversation; we add a small linear head on top of pooled
hidden states.

Two viable implementations (pick one when filling in):

  A) Frozen-base + tiny head: take the base model's last-layer mean-
     pooled hidden state on the global-state prompt, run a 2-layer
     MLP to scalar. Cheap, fast.

  B) LoRA-tuned base + head: add a fourth LoRA adapter (`critic`),
     train head AND adapter. More expressive, more compute.

A is recommended for the first MAPPO iteration; switch to B if the
critic is the bottleneck.
"""

from __future__ import annotations

from typing import List, Dict


class CentralizedValueNet:
    """Maps a global-state representation to V(s) ∈ ℝ.

    Global state = the concatenated conversation context the agents see
    (chat history + latest user message + coord analysis). We feed it
    through the base model in a no-grad pass (or with the dedicated
    `critic` LoRA adapter active), pool the last-layer hidden states,
    and apply a small MLP head.

    Args:
        base_model: shared base model (same one MultiAgentPolicy uses).
        tokenizer: shared tokenizer.
        hidden_dim: base model's hidden size (e.g., 4096 for Llama-3-8B).
        head_hidden: size of the MLP intermediate layer.
        device: cuda device for the head.
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        hidden_dim: int = 4096,
        head_hidden: int = 512,
        device: str = "cuda:0",
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.device = device
        # TODO:
        #   - self.head = nn.Sequential(
        #         nn.Linear(hidden_dim, head_hidden),
        #         nn.GELU(),
        #         nn.Linear(head_hidden, 1),
        #     ).to(device).to(torch_dtype)
        self.head = None

    def __call__(self, global_state_text: str) -> float:
        """Compute V(s) for a global state string.

        TODO:
          - tokenize global_state_text
          - forward through base_model with output_hidden_states=True
          - mean-pool last-layer hidden states across the sequence
            (or take last-token, depending on what works better — try
            mean-pool first since the value head should integrate over
            the whole context)
          - pass through self.head
          - return scalar
        """
        raise NotImplementedError

    def batched(self, global_states: List[str]) -> List[float]:
        """Vectorized version for the trainer's update loop."""
        raise NotImplementedError

    def trainable_parameters(self):
        """Yield only the head parameters (and `critic` adapter params
        if option B is chosen)."""
        raise NotImplementedError


def encode_global_state(
    chat_history: List[Dict[str, str]],
    user_message: str,
    coord_analysis: dict,
) -> str:
    """Concatenate everything the value head needs to see.

    Per proposal: "global state s_t (concatenated context of all
    agents)". We include the chat history, the latest user message,
    and the coordinator's analysis — i.e., everything visible to the
    triad before the therapist drafts a response.

    Returns a single string that the value net's tokenizer encodes.
    """
    raise NotImplementedError
