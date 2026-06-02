#!/usr/bin/env python3
"""MAPPO training entrypoint for MindSafe.

Loop:
  1. Start frozen 70B AWQ judge (vLLM HTTP, GPUs 1-2).
  2. Build MultiAgentPolicy (Llama-3-8B + 3 LoRA adapters, GPU 0).
  3. Build CentralizedValueNet (small head on the same base).
  4. Build MAPPOTrainer.
  5. For total_episodes // n_episodes_per_update updates:
       a. collect_rollouts → buffer
       b. compute_advantages(buffer)
       c. trainer.update(buffer)
       d. log
       e. periodic checkpoint
       f. periodic baseline-style eval
  6. Final checkpoint + eval, tear down judge.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

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
from src.mappo.rollout import _FrozenBaseClient
from src.mappo.trainer import MAPPOConfig
from src.models import (
    judge_client_from_config,
    load_config,
    start_judge_server,
)
from src.simulation import PsiPatientSimulator


def _patient_factory(base_model, tokenizer, device):
    """Returns a function (scenario, conv_idx, base_seed) -> patient simulator.

    The patient uses the shared base model with NO adapter active —
    saves a whole GPU's worth of memory vs. running a separate frozen
    LLaMA for the simulator.
    """
    client = _FrozenBaseClient(base_model, tokenizer, device=device)

    def make(scenario: str, conv_idx: int, base_seed: int):
        return PsiPatientSimulator(
            llm_client=client,
            scenario_name=scenario,
            seed=base_seed + conv_idx,
            conv_idx=conv_idx,
            base_seed=base_seed,
        )
    return make


async def main_async(config: dict, out_dir: Path, args):
    print(f"[mappo] Starting vLLM judge server (GPUs {config['judge_model']['server']['gpu_ids']})")
    server = start_judge_server(config["judge_model"])

    try:
        judge_client = judge_client_from_config(config["judge_model"])
        consensus_metrics = ConsensusMetrics()

        print(f"[mappo] Loading policy: {config['mas_model']['model_id']}")
        policy = MultiAgentPolicy(
            base_model_id=config["mas_model"]["model_id"],
            agent_configs=config["agents"],
            lora=LoRAConfigSpec(**{
                "rank": config["mappo"]["lora"]["rank"],
                "alpha": config["mappo"]["lora"]["alpha"],
                "dropout": config["mappo"]["lora"]["dropout"],
                "target_modules": tuple(config["mappo"]["lora"]["target_modules"]),
            }),
            device=config["mas_model"]["device"],
            torch_dtype=config["mas_model"]["torch_dtype"],
        )
        print(f"[mappo] Trainable LoRA params: {policy.n_trainable_params():,}")

        value_net = CentralizedValueNet(
            base_model=policy.base_model,
            tokenizer=policy.tokenizer,
            device=config["mas_model"]["device"],
        )

        mp_cfg = MAPPOConfig(
            clip_eps=config["mappo"]["ppo"]["clip_eps"],
            n_epochs_per_update=config["mappo"]["ppo"]["n_epochs_per_update"],
            minibatch_size=config["mappo"]["ppo"]["minibatch_size"],
            lr_policy=config["mappo"]["ppo"]["lr_policy"],
            lr_value=config["mappo"]["ppo"]["lr_value"],
            grad_clip=config["mappo"]["ppo"]["grad_clip"],
            entropy_coef=config["mappo"]["ppo"]["entropy_coef"],
            value_coef=config["mappo"]["ppo"]["value_coef"],
            beta=config["mappo"]["reward"]["beta"],
            tau=config["mappo"]["reward"]["tau"],
        )
        trainer = MAPPOTrainer(policy, value_net, mp_cfg)

        start_update = 0
        if args.resume_from:
            print(f"[mappo] Resuming from {args.resume_from}")
            resumed_idx = trainer.load_checkpoint(Path(args.resume_from))
            if resumed_idx >= 0:
                start_update = resumed_idx + 1
                print(f"[mappo] Resumed at update_idx={resumed_idx}; "
                      f"next update will be {start_update}.")

        make_patient = _patient_factory(
            policy.base_model, policy.tokenizer, config["mas_model"]["device"],
        )

        # ---- main loop ----
        n_eps_per_scen_per_update = config["mappo"]["n_episodes_per_scenario_per_update"]
        eps_per_update = n_eps_per_scen_per_update * len(config["scenarios"])
        n_updates = config["mappo"]["total_episodes"] // eps_per_update

        save_every = config["mappo"]["save_every"]
        eval_every = config["mappo"]["eval_every"]

        log_path = out_dir / "train_log.jsonl"
        with log_path.open("a") as log_f:
            for update_idx in range(start_update, n_updates):
                buffer = await collect_rollouts(
                    policy=policy,
                    value_net=value_net,
                    patient_factory=make_patient,
                    scenarios=config["scenarios"],
                    n_episodes_per_scenario=n_eps_per_scen_per_update,
                    max_turns=config["max_turns"],
                    judge_client=judge_client,
                    consensus_metrics=consensus_metrics,
                    hook=None,
                    beta=config["mappo"]["reward"]["beta"],
                    tau=config["mappo"]["reward"]["tau"],
                    alpha_quality=config["mappo"]["reward"]["alpha_quality"],
                    base_seed=config["seed"] + update_idx * 1000,
                )
                compute_advantages(
                    buffer,
                    gamma=config["mappo"]["gamma"],
                    lam=config["mappo"]["lam"],
                )
                stats = trainer.update(buffer)
                roll = buffer.per_arm_summary()
                row = {"update": update_idx, **roll, **stats}
                log_f.write(json.dumps(row, default=str) + "\n")
                log_f.flush()
                print(
                    f"[update {update_idx:04d}] "
                    f"r_t={roll.get('mean_r_therapy', 0):.3f} "
                    f"c={roll.get('mean_c_consensus', 0):.3f} "
                    f"R={roll.get('mean_r_joint', 0):.3f} "
                    f"σ={roll.get('mean_sigma', 0):.3f} "
                    f"Vloss={stats.get('value_loss', 0):.4f} "
                    f"clip={stats.get('clip_frac', 0):.2%}"
                )

                if (update_idx + 1) % save_every == 0:
                    ck = out_dir / f"ckpt_{update_idx:05d}"
                    trainer.save_checkpoint(ck, update_idx=update_idx)
                    print(f"[mappo] saved {ck}")

                if (update_idx + 1) % eval_every == 0:
                    print(f"[mappo] running eval @ update {update_idx}")
                    turns_path = out_dir / f"eval_{update_idx:05d}_turns.jsonl"
                    report = await evaluate_against_baseline(
                        policy=policy,
                        judge_client=judge_client,
                        consensus_metrics=consensus_metrics,
                        scenarios=config["scenarios"],
                        n_eps_per_scenario=config["mappo"]["eval_n_eps_per_scenario"],
                        max_turns=config["max_turns"],
                        distance_threshold=0.07,
                        safety_threshold=0.7,
                        tau=config["mappo"]["reward"]["tau"],
                        base_seed=10_000 + update_idx,
                        turns_out_path=turns_path,
                    )
                    save_eval_report(report, out_dir / f"eval_{update_idx:05d}.json")
                    print(
                        f"[eval] mean_c={report.get('mean_c_consensus', 0):.3f} "
                        f"σ={report.get('mean_external_safety', 0):.3f} "
                        f"asr={report.get('asr', 0):.1%} "
                        f"detect={report.get('detection_rate', 0):.1%} "
                        f"fpr={report.get('fpr', 0):.1%}"
                    )

        # ---- final ----
        trainer.save_checkpoint(out_dir / "final", update_idx=n_updates - 1)
        print("[mappo] done.")

    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mappo_4gpu.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint dir to resume from.")
    parser.add_argument("--total-episodes", type=int, default=None,
                        help="Override config.mappo.total_episodes.")
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Override config.scenarios. Useful for smoke tests "
             "(--scenarios ambiguous_crisis).",
    )
    parser.add_argument(
        "--n-eps-per-scen-per-update", type=int, default=None,
        help="Override config.mappo.n_episodes_per_scenario_per_update.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Override config.max_turns. Cuts episode length for smoke tests.",
    )
    parser.add_argument(
        "--save-every", type=int, default=None,
        help="Override config.mappo.save_every.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=None,
        help="Override config.mappo.eval_every.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.total_episodes is not None:
        config["mappo"]["total_episodes"] = args.total_episodes
    if args.scenarios:
        config["scenarios"] = args.scenarios
    if args.n_eps_per_scen_per_update is not None:
        config["mappo"]["n_episodes_per_scenario_per_update"] = args.n_eps_per_scen_per_update
    if args.max_turns is not None:
        config["max_turns"] = args.max_turns
    if args.save_every is not None:
        config["mappo"]["save_every"] = args.save_every
    if args.eval_every is not None:
        config["mappo"]["eval_every"] = args.eval_every

    # Sanity-check the loop will actually run at least once
    eps_per_update = (
        config["mappo"]["n_episodes_per_scenario_per_update"]
        * len(config["scenarios"])
    )
    if config["mappo"]["total_episodes"] < eps_per_update:
        raise SystemExit(
            f"total_episodes ({config['mappo']['total_episodes']}) is smaller "
            f"than one update batch ({eps_per_update} = "
            f"n_eps_per_scen_per_update * n_scenarios). Reduce one or both."
        )

    out_dir = Path(args.output_dir or
                   f"data/results/mappo/run_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"[mappo] Output dir: {out_dir}")
    print(f"[mappo] Scenarios: {config['scenarios']}")
    print(f"[mappo] eps/update: {eps_per_update} | "
          f"n_updates: {config['mappo']['total_episodes'] // eps_per_update}")
    asyncio.run(main_async(config, out_dir, args))


if __name__ == "__main__":
    main()
