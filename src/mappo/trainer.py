"""MAPPO update loop — clipped PPO objective per agent, shared advantage.

Implements proposal Eq. 5:

    L_CLIP_i(θ_i) = E_t[ min( ρ_t · Â_t,  clip(ρ_t, 1−ε, 1+ε) · Â_t ) ]

with ρ_t = exp(log π_θ_i(a_t^i | τ_t^i) − log π_θ_i_old(a_t^i | τ_t^i))
and Â_t computed centrally over the joint reward.

Per-turn there are FOUR action records (one per agent role):
  - coord_analyze  → trains the coordinator adapter
  - therapist      → trains the therapist adapter
  - monitor        → trains the monitor adapter
  - coord_route    → trains the coordinator adapter (same one again)

So the coordinator adapter receives gradient from BOTH analyze and
route prompts (Q2 default = yes).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .policy import MultiAgentPolicy
from .rollout import RolloutBuffer, TurnRecord
from .value_net import CentralizedValueNet


@dataclass
class MAPPOConfig:
    clip_eps: float = 0.2
    n_epochs_per_update: int = 4
    minibatch_size: int = 32        # turns per minibatch
    lr_policy: float = 1e-5
    lr_value: float = 1e-4
    grad_clip: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    beta: float = 1.0     # mirror of reward.beta — informational
    tau: float = 0.1      # mirror of reward.tau  — informational


# Maps role → which agent's adapter to use for the log-prob recompute.
_ROLE_TO_AGENT = {
    "coord_analyze": "coordinator",
    "therapist":     "therapist",
    "monitor":       "monitor",
    "coord_route":   "coordinator",
}


class MAPPOTrainer:
    def __init__(
        self,
        policy: MultiAgentPolicy,
        value_net: CentralizedValueNet,
        cfg: MAPPOConfig,
    ):
        self.policy = policy
        self.value_net = value_net
        self.cfg = cfg

        self.policy_optim = torch.optim.AdamW(
            list(self.policy.trainable_parameters()),
            lr=cfg.lr_policy,
        )
        self.value_optim = torch.optim.AdamW(
            list(self.value_net.trainable_parameters()),
            lr=cfg.lr_value,
        )

    # -----------------------------------------------------------------
    # Public: one PPO update over the buffer
    # -----------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if buffer.advantages is None or buffer.returns is None:
            raise RuntimeError("Call rollout.compute_advantages(buffer) before update().")

        steps: List[TurnRecord] = buffer.flatten_steps()
        n = len(steps)
        if n == 0:
            return {"n_steps": 0}

        # Normalize advantages — standard PPO trick, large variance
        # reduction without changing the optimum.
        adv = torch.tensor(buffer.advantages, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        rets = torch.tensor(buffer.returns, dtype=torch.float32)

        log = {
            "n_steps": n,
            "policy_loss/coord_analyze": 0.0,
            "policy_loss/therapist":     0.0,
            "policy_loss/monitor":       0.0,
            "policy_loss/coord_route":   0.0,
            "value_loss": 0.0,
            "kl/coord_analyze": 0.0,
            "kl/therapist":     0.0,
            "kl/monitor":       0.0,
            "kl/coord_route":   0.0,
            "clip_frac":        0.0,
        }
        n_minibatch_steps = 0
        clip_count = 0
        clip_total = 0

        idx_all = list(range(n))
        for epoch in range(self.cfg.n_epochs_per_update):
            random.shuffle(idx_all)
            for start in range(0, n, self.cfg.minibatch_size):
                idxs = idx_all[start : start + self.cfg.minibatch_size]
                batch_steps = [steps[i] for i in idxs]
                batch_adv = adv[idxs]
                batch_ret = rets[idxs]

                # ---- per-agent policy losses (centralized advantage) ----
                self.policy_optim.zero_grad(set_to_none=True)
                total_policy_loss = torch.zeros((), device=self.policy.device)
                for role, agent_name in _ROLE_TO_AGENT.items():
                    agent = getattr(self.policy, agent_name)
                    pl, kl_, cf, ct = self._policy_loss_for_role(
                        agent, batch_steps, batch_adv, role,
                    )
                    total_policy_loss = total_policy_loss + pl
                    log[f"policy_loss/{role}"] += float(pl.detach().cpu())
                    log[f"kl/{role}"] += float(kl_)
                    clip_count += cf
                    clip_total += ct

                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.trainable_parameters()), self.cfg.grad_clip
                )
                self.policy_optim.step()

                # ---- value loss (separate optimizer) -------------------
                self.value_optim.zero_grad(set_to_none=True)
                gst_batch = [s.global_state_text for s in batch_steps]
                v_pred = self.value_net.batched(gst_batch)            # [B]
                value_loss = ((v_pred.float() - batch_ret.to(v_pred.device).float()) ** 2).mean()
                (self.cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.value_net.trainable_parameters()), self.cfg.grad_clip
                )
                self.value_optim.step()

                log["value_loss"] += float(value_loss.detach().cpu())
                n_minibatch_steps += 1

        # average across minibatches
        if n_minibatch_steps > 0:
            for k in list(log.keys()):
                if k.startswith(("policy_loss/", "kl/", "value_loss")):
                    log[k] = log[k] / n_minibatch_steps
        if clip_total > 0:
            log["clip_frac"] = clip_count / clip_total
        return log

    # -----------------------------------------------------------------
    # Internal: per-role policy loss for one minibatch
    # -----------------------------------------------------------------

    def _policy_loss_for_role(self, agent, batch_steps, batch_adv, role):
        """Sum the clipped surrogate over all turns in this minibatch
        for a single role's action records.

        Each turn has one action record per role; we recompute log-probs
        under the CURRENT adapter, compare to the old (sample-time) ones,
        and apply the PPO-clip objective.
        """
        cum_loss = torch.zeros((), device=self.policy.device)
        kl_acc, n_tok_acc = 0.0, 0
        clipped, total = 0, 0

        for step, A_scalar in zip(batch_steps, batch_adv):
            act = step.actions.get(role)
            if act is None or act["response_ids"].size == 0:
                continue
            prompt_ids   = torch.from_numpy(act["prompt_ids"]).long()
            response_ids = torch.from_numpy(act["response_ids"]).long()
            old_lp       = torch.from_numpy(act["old_log_probs"]).float().to(self.policy.device)

            new_lp = agent.compute_log_probs(prompt_ids, response_ids)  # [n_resp], grad enabled
            # Token-mean (could also be sum; mean keeps loss scale token-count invariant).
            ratio = torch.exp(new_lp - old_lp.detach())
            A = A_scalar.detach().to(ratio.device).to(ratio.dtype)
            unclipped = ratio * A
            clipped_r = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * A
            surrogate = -torch.min(unclipped, clipped_r).mean()
            cum_loss = cum_loss + surrogate

            # KL diagnostic (rough — token-level mean)
            with torch.no_grad():
                kl = (old_lp.detach() - new_lp.detach()).mean().item()
                kl_acc += kl
                n_tok_acc += 1
                # Clip fraction: how often the clip kicked in
                clip_mask = ((ratio < 1 - self.cfg.clip_eps) | (ratio > 1 + self.cfg.clip_eps))
                clipped += int(clip_mask.sum().item())
                total += int(clip_mask.numel())

        kl_mean = (kl_acc / n_tok_acc) if n_tok_acc > 0 else 0.0
        return cum_loss / max(1, len(batch_steps)), kl_mean, clipped, total

    # -----------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------

    def save_checkpoint(self, dir_path: Path) -> None:
        dir_path = Path(dir_path); dir_path.mkdir(parents=True, exist_ok=True)
        self.policy.save(dir_path / "policy")
        self.value_net.save(dir_path / "value")
        torch.save({
            "policy_optim": self.policy_optim.state_dict(),
            "value_optim":  self.value_optim.state_dict(),
        }, dir_path / "optim.pt")

    def load_checkpoint(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        self.policy.load(dir_path / "policy")
        self.value_net.load(dir_path / "value")
        opt = torch.load(dir_path / "optim.pt", map_location="cpu")
        self.policy_optim.load_state_dict(opt["policy_optim"])
        self.value_optim.load_state_dict(opt["value_optim"])
