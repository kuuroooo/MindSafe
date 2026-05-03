#!/usr/bin/env python3
"""Compute and analyze the unsafe-consensus penalty c_consensus per turn.

Implements proposal Eq. 4:

    c_consensus(s_t) = exp(-d(h_therapist, h_monitor) / τ) · (1 - σ(s_t))

This is the continuous reward-shaping signal MAPPO will minimize during
training. Before training, we use this script to validate three things
the penalty needs to satisfy:

  1. Dynamic range — c spans a wide enough interval to provide gradient.
  2. Validity — high-c turns are clinically problematic (sanity check).
  3. Responsiveness — attacks shift the c distribution (cross-arm comparison).

Usage:
  python scripts/consensus_penalty.py <sweep_dir> [<sweep_dir> ...] \
      [--tau 0.1] [--top-k 3] [--by-scenario] [--released-only]

Examples:
  # Distribution + cross-arm comparison
  python scripts/consensus_penalty.py \
    data/results/baseline/sweep_20260425_165007 \
    data/results/baseline/run_22355364

  # Just the high-penalty turns released as safe (the false-positive set
  # MAPPO is supposed to suppress).
  python scripts/consensus_penalty.py \
    data/results/baseline/sweep_20260425_165007 \
    data/results/baseline/run_22355364 \
    --released-only --top-k 5
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Tuple


def consensus_penalty(d: float, sigma: float, tau: float) -> float:
    """c_consensus = exp(-d / tau) · (1 - sigma).

    High when agents are in latent agreement (small d → exp ≈ 1) and the
    response is judged unsafe (small sigma → 1-sigma ≈ 1).
    """
    return math.exp(-d / tau) * (1.0 - sigma)


def quantiles(xs, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    s = sorted(xs)
    n = len(s)
    return [s[min(n - 1, int(q * n))] for q in qs]


def walk_arm(arm_dir: Path) -> Iterator[Tuple[str, str, dict]]:
    """Yield (scenario, conv_id, turn_dict) for every turn in an arm."""
    logs = arm_dir / "logs"
    if not logs.exists():
        return
    for scen_dir in sorted(logs.iterdir()):
        if not scen_dir.is_dir():
            continue
        for cp in sorted(scen_dir.glob("conv_*.jsonl")):
            if cp.name.endswith(".v2.jsonl") or cp.name.endswith(".repaired.jsonl"):
                continue
            for line in cp.read_text().splitlines():
                if line.strip():
                    yield scen_dir.name, cp.stem, json.loads(line)


def gather_arms(sweep_dirs):
    """Yield (arm_label, arm_dir) from a list of sweep dirs."""
    for sweep in sweep_dirs:
        sweep = Path(sweep)
        if not sweep.exists():
            print(f"  WARNING: {sweep} not found, skipping")
            continue
        for arm_dir in sorted(d for d in sweep.iterdir() if d.is_dir()):
            yield arm_dir.name, arm_dir


def collect(arm_dir: Path, tau: float, released_only: bool):
    """Return list of dicts with c, sigma, d, scenario, conv, turn_dict."""
    rows = []
    for scen, conv, t in walk_arm(arm_dir):
        d = t.get("latent_distance")
        sigma = t.get("external_safety")
        if d is None or sigma is None:
            continue
        coord = str((t.get("coordinator_final") or {}).get("final_label", "safe")).lower()
        if released_only and coord != "safe":
            continue
        rows.append({
            "c": consensus_penalty(d, sigma, tau),
            "sigma": sigma,
            "d": d,
            "scenario": scen,
            "conv": conv,
            "turn": t.get("turn"),
            "coord": coord,
            "user": (t.get("user_message") or "")[:120],
            "resp": (t.get("response") or "")[:120],
        })
    return rows


def fmt_dist(rows, label, baseline_p95=None):
    cs = [r["c"] for r in rows]
    if not cs:
        return f"{label:<38} | no data"
    qs = quantiles(cs)
    line = (f"{label:<38} | n={len(cs):>4d} | "
            f"mean={statistics.mean(cs):.3f} | p50={qs[0]:.3f} | "
            f"p75={qs[1]:.3f} | p90={qs[2]:.3f} | p95={qs[3]:.3f} | "
            f"p99={qs[4]:.3f} | max={max(cs):.3f}")
    if baseline_p95 is not None:
        above = sum(1 for c in cs if c > baseline_p95) / len(cs)
        line += f"   |   {above:.1%} > psi p95"
    return line


def render_top(rows, top_k):
    rows = sorted(rows, key=lambda r: -r["c"])[:top_k]
    out = []
    for r in rows:
        out.append(
            f"  c={r['c']:.3f}  σ={r['sigma']:.2f}  d={r['d']:.3f}  "
            f"coord={r['coord']:<6}  {r['scenario']}/{r['conv']}/turn={r['turn']}"
        )
        out.append(f"     user: {r['user']!r}")
        out.append(f"     resp: {r['resp']!r}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep_dirs", nargs="+",
                   help="One or more sweep dirs to walk.")
    p.add_argument("--tau", type=float, default=0.1,
                   help="Temperature for the latent-distance kernel "
                        "(default 0.1; smaller = sharper). Eq. 4.")
    p.add_argument("--top-k", type=int, default=3,
                   help="Show top-K turns per arm by c_consensus.")
    p.add_argument("--by-scenario", action="store_true",
                   help="Print per-arm × per-scenario distribution table.")
    p.add_argument("--released-only", action="store_true",
                   help="Restrict analysis to turns where coord released "
                        "as safe (the false-positive set MAPPO targets).")
    args = p.parse_args()

    print(f"\n c_consensus = exp(-d/τ) · (1 - σ),    τ = {args.tau}"
          + ("  [released_safe only]" if args.released_only else "")
          + "\n")

    arms = list(gather_arms(args.sweep_dirs))

    # Collect per-arm rows
    per_arm = {}
    for arm_label, arm_dir in arms:
        rows = collect(arm_dir, args.tau, args.released_only)
        if rows:
            per_arm[arm_label] = rows

    if not per_arm:
        raise SystemExit("No data collected.")

    # Distribution headline
    print("=" * 130)
    print("PER-ARM DISTRIBUTION")
    print("=" * 130)
    base = per_arm.get("psi")
    base_p95 = quantiles([r["c"] for r in base])[3] if base else None
    for arm_label, rows in per_arm.items():
        print(fmt_dist(rows, arm_label, baseline_p95=base_p95))

    # Cross-arm shifts
    if base:
        print()
        print("CROSS-ARM SHIFT (vs psi baseline):")
        bcs = [r["c"] for r in base]
        bmean = statistics.mean(bcs); bp95 = base_p95
        for arm_label, rows in per_arm.items():
            if arm_label == "psi":
                continue
            cs = [r["c"] for r in rows]
            mdelta = statistics.mean(cs) - bmean
            p95 = quantiles(cs)[3]
            above = sum(1 for c in cs if c > bp95) / len(cs)
            print(f"  {arm_label:<38}  mean Δ={mdelta:+.3f}   "
                  f"p95 Δ={p95-bp95:+.3f}   tail>{bp95:.3f} = {above:.1%}")

    # Top-K turns per arm
    print()
    print("=" * 130)
    print(f"TOP-{args.top_k} c_consensus TURNS PER ARM (validity check)")
    print("=" * 130)
    for arm_label, rows in per_arm.items():
        print(f"\n=== {arm_label} ===")
        print(render_top(rows, args.top_k))

    # Per-scenario × arm
    if args.by_scenario:
        print()
        print("=" * 130)
        print("PER-SCENARIO MEAN c_consensus")
        print("=" * 130)
        scens = sorted({r["scenario"] for rows in per_arm.values() for r in rows})
        header = f"{'scenario':<24} | " + ' | '.join(f'{a:>22}' for a in per_arm)
        print(header)
        print("-" * len(header))
        for scen in scens:
            cells = []
            for arm_label, rows in per_arm.items():
                cs = [r["c"] for r in rows if r["scenario"] == scen]
                if cs:
                    cells.append(f"{statistics.mean(cs):>11.3f}  (n={len(cs):>3d})")
                else:
                    cells.append(f"{'—':>22}")
            print(f"{scen:<24} | " + ' | '.join(cells))


if __name__ == "__main__":
    main()
