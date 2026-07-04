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
    minibatch_size: int = 32
    lr_policy: float = 1e-5
    lr_value: float = 1e-4
    grad_clip: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    beta: float = 1.0
    tau: float = 0.1


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


    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if buffer.advantages is None or buffer.returns is None:
            raise RuntimeError("Call rollout.compute_advantages(buffer) before update().")

        steps: List[TurnRecord] = buffer.flatten_steps()
        n = len(steps)
        if n == 0:
            return {"n_steps": 0}

        # diagnostics on raw (pre-normalization) advantages/returns — what the critic actually optimized
        raw_adv  = np.asarray(buffer.advantages, dtype=np.float64)
        raw_rets = np.asarray(buffer.returns,    dtype=np.float64)
        rollout_values = np.array([s.value for s in steps], dtype=np.float64)
        # explained variance of the critic: 1=perfect, 0=predicts mean, <0=worse than mean (broken)
        ret_var = float(np.var(raw_rets))
        if ret_var > 1e-12:
            explained_var = 1.0 - float(np.var(raw_rets - rollout_values)) / ret_var
        else:
            explained_var = float("nan")
        r_therapy_arr = np.array([s.r_therapy   for s in steps], dtype=np.float64)
        c_cons_arr    = np.array([s.c_consensus for s in steps], dtype=np.float64)
        beta = self.cfg.beta
        diag_pre = {
            "explained_variance": explained_var,
            "adv_mean_raw":       float(np.mean(raw_adv)),
            "adv_std_raw":        float(np.std(raw_adv)),
            "returns_mean":       float(np.mean(raw_rets)),
            "returns_std":        float(np.std(raw_rets)),
            "values_mean":        float(np.mean(rollout_values)),
            "values_std":         float(np.std(rollout_values)),
            "r_therapy_std":      float(np.std(r_therapy_arr)),
            "c_consensus_std":    float(np.std(c_cons_arr)),
            "beta_c_consensus_std": beta * float(np.std(c_cons_arr)),
            "advantage_normalized": True,
        }

        adv = torch.tensor(buffer.advantages, dtype=torch.float32)
        # standard ppo advantage normalization: cuts gradient variance without moving the optimum
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
            **diag_pre,
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

                self.policy_optim.zero_grad(set_to_none=True)
                n_units = max(1, len(batch_steps)) * len(_ROLE_TO_AGENT)
                for role, agent_name in _ROLE_TO_AGENT.items():
                    agent = getattr(self.policy, agent_name)
                    pl_val, kl_, cf, ct = self._policy_loss_for_role(
                        agent, batch_steps, batch_adv, role,
                        n_units=n_units,
                    )
                    log[f"policy_loss/{role}"] += pl_val
                    log[f"kl/{role}"] += kl_
                    clip_count += cf
                    clip_total += ct

                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.trainable_parameters()), self.cfg.grad_clip
                )
                self.policy_optim.step()

                self.value_optim.zero_grad(set_to_none=True)
                gst_batch = [s.global_state_text for s in batch_steps]
                v_pred = self.value_net.batched(gst_batch)
                value_loss = ((v_pred.float() - batch_ret.to(v_pred.device).float()) ** 2).mean()
                (self.cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.value_net.trainable_parameters()), self.cfg.grad_clip
                )
                self.value_optim.step()

                log["value_loss"] += float(value_loss.detach().cpu())
                n_minibatch_steps += 1

        if n_minibatch_steps > 0:
            for k in list(log.keys()):
                if k.startswith(("policy_loss/", "kl/", "value_loss")):
                    log[k] = log[k] / n_minibatch_steps
        if clip_total > 0:
            log["clip_frac"] = clip_count / clip_total
        return log


    def _policy_loss_for_role(self, agent, batch_steps, batch_adv, role, n_units):
        cum_loss_value = 0.0
        kl_acc, n_tok_acc = 0.0, 0
        clipped, total = 0, 0
        n_seen = 0

        for step, A_scalar in zip(batch_steps, batch_adv):
            act = step.actions.get(role)
            if act is None or act["response_ids"].size == 0:
                continue
            prompt_ids   = torch.from_numpy(act["prompt_ids"]).long()
            response_ids = torch.from_numpy(act["response_ids"]).long()
            old_lp       = torch.from_numpy(act["old_log_probs"]).float().to(self.policy.device)

            new_lp = agent.compute_log_probs(prompt_ids, response_ids)
            ratio = torch.exp(new_lp - old_lp.detach())
            A = A_scalar.detach().to(ratio.device).to(ratio.dtype)
            unclipped = ratio * A
            clipped_r = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * A
            surrogate = -torch.min(unclipped, clipped_r).mean()

            # backprop per step to bound peak activation memory; /n_units so accumulated grads == averaged loss
            (surrogate / n_units).backward()
            cum_loss_value += float(surrogate.detach().cpu())
            n_seen += 1

            with torch.no_grad():
                kl = (old_lp.detach() - new_lp.detach()).mean().item()
                kl_acc += kl
                n_tok_acc += 1
                clip_mask = ((ratio < 1 - self.cfg.clip_eps) | (ratio > 1 + self.cfg.clip_eps))
                clipped += int(clip_mask.sum().item())
                total += int(clip_mask.numel())
            del new_lp, ratio, surrogate, unclipped, clipped_r

        avg_loss = cum_loss_value / max(1, n_seen)
        kl_mean = (kl_acc / n_tok_acc) if n_tok_acc > 0 else 0.0
        return avg_loss, kl_mean, clipped, total


    def save_checkpoint(self, dir_path: Path, update_idx: int = -1) -> None:
        dir_path = Path(dir_path); dir_path.mkdir(parents=True, exist_ok=True)
        self.policy.save(dir_path / "policy")
        self.value_net.save(dir_path / "value")
        torch.save({
            "policy_optim": self.policy_optim.state_dict(),
            "value_optim":  self.value_optim.state_dict(),
            "update_idx":   update_idx,
        }, dir_path / "optim.pt")

    def load_checkpoint(self, dir_path: Path) -> int:
        dir_path = Path(dir_path)
        self.policy.load(dir_path / "policy")
        self.value_net.load(dir_path / "value")
        opt = torch.load(dir_path / "optim.pt", map_location="cpu")
        self.policy_optim.load_state_dict(opt["policy_optim"])
        self.value_optim.load_state_dict(opt["value_optim"])
        return int(opt.get("update_idx", -1))
