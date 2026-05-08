#!/usr/bin/env python3
"""MAPPO training entrypoint for MindSafe.

Top-level loop:

    1. Load config (configs/mappo_4gpu.yaml).
    2. Start the frozen 70B AWQ judge (vLLM HTTP, GPUs 1-2).
    3. Construct MultiAgentPolicy (LoRA adapters on Llama-3-8B, GPU 0).
    4. Construct CentralizedValueNet (small head on the same base).
    5. Construct MAPPOTrainer.
    6. For total_episodes // n_episodes_per_update updates:
         a. collect_rollouts(...)  → buffer
         b. compute_advantages(buffer, gamma, lam)
         c. trainer.update(buffer)
         d. log
         e. every save_every: trainer.save_checkpoint(...)
         f. every eval_every: eval.evaluate_against_baseline(...)
    7. Final checkpoint + eval, tear down judge.

This is a scaffold — `raise NotImplementedError` for the actual loop
until policy/rollout/trainer are implemented.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import ConsensusMetrics
from src.mappo import (
    CentralizedValueNet,
    MAPPOTrainer,
    MultiAgentPolicy,
    collect_rollouts,
    compute_advantages,
)
from src.mappo.eval import evaluate_against_baseline, save_eval_report
from src.mappo.policy import LoRAConfigSpec
from src.mappo.trainer import MAPPOConfig
from src.models import (
    judge_client_from_config,
    load_config,
    start_judge_server,
)
from src.simulation import PsiPatientSimulator


def _patient_factory(mas_client_or_none):
    """Returns a function (scenario, conv_idx, base_seed) -> patient.

    Patient simulator uses an LLM client for message generation. During
    training we reuse the base policy model for that (no separate
    generator — saves GPU memory). Pass None or the base model.
    """
    def make(scenario: str, conv_idx: int, base_seed: int):
        return PsiPatientSimulator(
            llm_client=mas_client_or_none,
            scenario_name=scenario,
            seed=base_seed + conv_idx,
            conv_idx=conv_idx,
            base_seed=base_seed,
        )
    return make


async def main_async(config: dict, out_dir: Path, args):
    raise NotImplementedError(
        "MAPPO training loop not yet implemented. "
        "Fill in src.mappo modules first per src/mappo/README.md."
    )

    # The shape it will take, once policy/rollout/trainer are real:
    #
    # 1. Start judge
    # server = start_judge_server(config["judge_model"])
    # judge_client = judge_client_from_config(config["judge_model"])
    # consensus_metrics = ConsensusMetrics()
    #
    # 2. Build policy + value net
    # policy = MultiAgentPolicy(
    #     base_model_id=config["mas_model"]["model_id"],
    #     agent_configs=config["agents"],
    #     lora=LoRAConfigSpec(**config["mappo"]["lora"]),
    #     device=config["mas_model"]["device"],
    # )
    # value_net = CentralizedValueNet(
    #     base_model=policy.base_model,
    #     tokenizer=policy.tokenizer,
    # )
    #
    # 3. Trainer
    # mp_cfg = MAPPOConfig(**config["mappo"]["ppo"])
    # trainer = MAPPOTrainer(policy, value_net, mp_cfg)
    #
    # 4. Patient factory
    # make_patient = _patient_factory(policy.base_model)
    #
    # 5. Loop
    # n_updates = config["mappo"]["total_episodes"] // config["mappo"]["n_episodes_per_update"]
    # for update_idx in range(n_updates):
    #     buffer = await collect_rollouts(
    #         policy=policy, value_net=value_net,
    #         patient_factory=make_patient,
    #         scenarios=config["scenarios"],
    #         n_episodes_per_scenario=config["mappo"]["n_episodes_per_scenario_per_update"],
    #         max_turns=config["max_turns"],
    #         judge_client=judge_client,
    #         consensus_metrics=consensus_metrics,
    #         hook=None,  # no adversarial training in v0
    #         beta=mp_cfg.beta, tau=mp_cfg.tau,
    #         base_seed=config["seed"] + update_idx * 1000,
    #     )
    #     compute_advantages(buffer, gamma=config["mappo"]["gamma"],
    #                                lam=config["mappo"]["lam"])
    #     stats = trainer.update(buffer)
    #     log(stats, update_idx)
    #
    #     if (update_idx + 1) % config["mappo"]["save_every"] == 0:
    #         trainer.save_checkpoint(out_dir / f"ckpt_{update_idx:05d}")
    #     if (update_idx + 1) % config["mappo"]["eval_every"] == 0:
    #         report = await evaluate_against_baseline(
    #             policy=policy, judge_client=judge_client,
    #             consensus_metrics=consensus_metrics,
    #             scenarios=config["scenarios"],
    #             n_eps_per_scenario=config["mappo"]["eval_n_eps_per_scenario"],
    #             max_turns=config["max_turns"],
    #         )
    #         save_eval_report(report, out_dir / f"eval_{update_idx:05d}.json")
    #
    # 6. Final
    # trainer.save_checkpoint(out_dir / "final")
    # server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mappo_4gpu.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint dir to resume from.")
    parser.add_argument("--total-episodes", type=int, default=None,
                        help="Override config.mappo.total_episodes.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.total_episodes:
        config["mappo"]["total_episodes"] = args.total_episodes

    out_dir = Path(args.output_dir or
                   f"data/results/mappo/run_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config for reproducibility
    import yaml
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"[mappo] Output dir: {out_dir}")
    asyncio.run(main_async(config, out_dir, args))


if __name__ == "__main__":
    main()
