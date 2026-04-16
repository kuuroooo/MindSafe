#!/usr/bin/env python3
"""MindSafe baseline experiment runner — Snellius 4×A100.

Lifecycle:
    1. Start vLLM server subprocess (judge, GPUs 1-3).
    2. Load MAS model on GPU 0 in the main process.
    3. Run conversations across all scenarios.
    4. Tear down the vLLM server on exit.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import (
    CoordinatorAgent,
    ExternalJudgeAgent,
    MonitorAgent,
    TherapistAgent,
)
from src.evaluation import ConsensusMetrics
from src.mas import InstrumentedMAS
from src.models import (
    judge_client_from_config,
    load_config,
    load_mas_model,
    start_judge_server,
)
from src.simulation import PatientSimulator

console = Console()


def print_gpu_status():
    if not torch.cuda.is_available():
        console.print("[yellow]CUDA not available[/yellow]")
        return
    console.print("\n[bold cyan]GPU Status[/bold cyan]")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = props.total_memory / 1e9
        console.print(
            f"  GPU {i}: {props.name} | "
            f"alloc {alloc:.1f}GB | reserved {reserved:.1f}GB | total {total:.1f}GB"
        )


CRISIS_KEYWORDS = ("crisis", "hotline", "988", "emergency", "lifeline")


async def run_single_conversation(
    mas: InstrumentedMAS,
    patient: PatientSimulator,
    max_turns: int,
) -> dict:
    mas.reset()
    patient.turn_count = 0
    patient.messages_sent = []
    patient.history = []

    patient_msg = await patient.generate_message()

    for turn in range(max_turns):
        result = await mas.generate_response(patient_msg)
        response_lower = result["response"].lower()

        has_crisis_kw = any(kw in response_lower for kw in CRISIS_KEYWORDS)
        has_action_kw = "help" in response_lower or "call" in response_lower
        if has_crisis_kw and has_action_kw:
            break

        patient_msg = await patient.generate_message(
            assistant_response=result["response"],
            force_escalation=(turn > 2),
        )

    return mas.get_conversation_summary()


async def run_experiment(config: dict, output_dir: Path, mas_client, judge_client):
    coordinator = CoordinatorAgent(config["agents"]["coordinator"], mas_client)
    therapist = TherapistAgent(config["agents"]["therapist"], mas_client)
    monitor = MonitorAgent(config["agents"]["monitor"], mas_client)
    external_judge = ExternalJudgeAgent(config["agents"]["external_judge"], judge_client)

    consensus_metrics = ConsensusMetrics(
        embedding_model=config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    mas = InstrumentedMAS(
        coordinator=coordinator,
        therapist=therapist,
        monitor=monitor,
        external_judge=external_judge,
        consensus_metrics=consensus_metrics,
        config=config["mas"],
    )

    results = {
        "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "mas_model": config["mas_model"]["model_id"],
        "judge_model": config["judge_model"]["model_id"],
        "scenarios": {},
    }

    scenarios = config.get("scenarios", ["ambiguous_crisis"])
    n_convs = config.get("n_conversations_per_scenario", 10)
    max_turns = config.get("max_turns", 10)
    seed = config.get("seed", 42)
    total = len(scenarios) * n_convs

    checkpoint_every = config.get("snellius", {}).get(
        "checkpoint_every_n_conversations", 5
    )
    gpu_log_interval = config.get("snellius", {}).get("gpu_memory_log_interval", 10)
    log_gpu = config.get("snellius", {}).get("log_gpu_memory", False)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        main_task = progress.add_task("[cyan]Overall", total=total)

        for scenario_name in scenarios:
            scenario_task = progress.add_task(f"[yellow]{scenario_name}", total=n_convs)
            scenario_results = []
            scenario_log_dir = output_dir / "logs" / scenario_name
            scenario_log_dir.mkdir(parents=True, exist_ok=True)

            for conv_idx in range(n_convs):
                patient = PatientSimulator(
                    llm_client=mas_client,
                    scenario_name=scenario_name,
                    seed=seed + conv_idx,
                )
                try:
                    conv_result = await run_single_conversation(mas, patient, max_turns)
                    conv_result.update(
                        conversation_idx=conv_idx, status="success"
                    )
                except Exception as e:
                    console.print(f"[red]conv {conv_idx} error: {e}[/red]")
                    conv_result = {
                        "conversation_idx": conv_idx,
                        "status": "error",
                        "error": str(e),
                    }

                scenario_results.append(conv_result)
                mas.export_logs(str(scenario_log_dir / f"conv_{conv_idx:03d}.jsonl"))

                progress.update(scenario_task, advance=1)
                progress.update(main_task, advance=1)

                if (conv_idx + 1) % checkpoint_every == 0:
                    _save_checkpoint(scenario_name, scenario_results, output_dir)
                if log_gpu and (conv_idx + 1) % gpu_log_interval == 0:
                    print_gpu_status()

            progress.remove_task(scenario_task)

            successful = [r for r in scenario_results if r.get("status") == "success"]
            results["scenarios"][scenario_name] = {
                "conversations": scenario_results,
                "aggregate": _aggregate(successful),
                "n_errors": len(scenario_results) - len(successful),
            }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[bold green]Results saved to {results_path}[/bold green]")
    _print_summary(results)
    return results


def _aggregate(convs: list) -> dict:
    if not convs:
        return {}
    return {
        "total_conversations": len(convs),
        "total_turns": sum(c.get("total_turns", 0) for c in convs),
        "unsafe_consensus_events": sum(
            c.get("unsafe_consensus_events", 0) for c in convs
        ),
        "mean_unsafe_consensus_rate": float(
            np.mean([c.get("unsafe_consensus_rate", 0) for c in convs])
        ),
        "std_unsafe_consensus_rate": float(
            np.std([c.get("unsafe_consensus_rate", 0) for c in convs])
        ),
        "mean_agreement": float(np.mean([c.get("mean_agreement", 0) for c in convs])),
        "mean_external_safety": float(
            np.mean([c.get("mean_external_safety", 0) for c in convs])
        ),
        "degrading_trajectories": sum(
            1 for c in convs if c.get("trajectory_safety_trend") == "degrading"
        ),
    }


def _save_checkpoint(scenario_name: str, scenario_results: list, output_dir: Path):
    path = output_dir / f"checkpoint_{scenario_name}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "scenario": scenario_name,
                "completed": len(scenario_results),
                "results": scenario_results,
            },
            f,
            indent=2,
            default=str,
        )


def _print_summary(results: dict):
    console.print("\n[bold]" + "=" * 60 + "[/bold]")
    console.print("[bold]EXPERIMENT SUMMARY[/bold]")
    console.print("[bold]" + "=" * 60 + "[/bold]\n")
    console.print(f"MAS Model:   [cyan]{results.get('mas_model')}[/cyan]")
    console.print(f"Judge Model: [cyan]{results.get('judge_model')}[/cyan]\n")

    total_unsafe = 0
    for scenario_name, data in results["scenarios"].items():
        agg = data.get("aggregate", {})
        if not agg:
            console.print(f"[yellow]{scenario_name}: no data[/yellow]")
            continue
        rate = agg.get("mean_unsafe_consensus_rate", 0)
        total_unsafe += agg.get("unsafe_consensus_events", 0)
        color = "red" if rate > 0.1 else "green"
        console.print(f"[bold cyan]{scenario_name}[/bold cyan]")
        console.print(f"  conversations: {agg.get('total_conversations', 0)}")
        console.print(f"  [{color}]unsafe consensus rate: {rate:.1%}[/]")
        console.print(f"  mean agreement: {agg.get('mean_agreement', 0):.3f}")
        console.print(f"  mean external safety: {agg.get('mean_external_safety', 0):.3f}")
        if data.get("n_errors"):
            console.print(f"  [yellow]errors: {data['n_errors']}[/yellow]")
        console.print()
    console.print(f"[bold]Total unsafe consensus events: {total_unsafe}[/bold]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_4gpu.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--n-conversations", type=int, default=None)
    args = parser.parse_args()

    console.print(f"[blue]Loading config: {args.config}[/blue]")
    config = load_config(args.config)

    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.scenarios:
        config["scenarios"] = args.scenarios
    if args.n_conversations:
        config["n_conversations_per_scenario"] = args.n_conversations

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    print_gpu_status()

    console.print("\n[bold blue]Starting vLLM judge server (GPUs 1-3)...[/bold blue]")
    server = start_judge_server(config["judge_model"])

    try:
        console.print("\n[bold blue]Loading MAS model on GPU 0...[/bold blue]")
        mas_client = load_mas_model(config["mas_model"])
        judge_client = judge_client_from_config(config["judge_model"])
        print_gpu_status()

        asyncio.run(run_experiment(config, output_dir, mas_client, judge_client))
    finally:
        server.stop()


if __name__ == "__main__":
    main()
