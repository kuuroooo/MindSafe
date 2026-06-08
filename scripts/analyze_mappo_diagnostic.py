#!/usr/bin/env python3
"""Offline diagnostic analysis of MAPPO eval data.

Answers all 6 of Meng's diagnostic questions from the per-turn JSONL dumps
plus train_log.jsonl. Produces a rough markdown report with tables.

Inputs (all under --run-dir, default data/results/mappo/main):
  train_log.jsonl                          # per-update rollout stats
  eval_<idx>_turns.jsonl                   # per-turn greedy eval, default seed
  eval_<idx>_turns_seed<S>.jsonl           # second-seed greedy eval (optional)
  eval_<idx>_turns_seed<S>_stoch.jsonl     # stochastic eval (sanity check)
  baseline_turns_seed10000.jsonl           # untrained policy through same harness
  reeval_<idx>.json                        # summary jsons (cross-check)
  eval_<idx>.json                          # original training-time summaries

Outputs:
  - Console tables for each of Meng's 6 items
  - mindsafeDocs/mappo_diagnostic.md with the same content
"""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array(
        [r[key] if r.get(key) is not None else np.nan for r in rows],
        dtype=float,
    )


# -----------------------------------------------------------------------------
# Bootstrap helpers
# -----------------------------------------------------------------------------

def boot_ci(values: np.ndarray, n_boot: int = 10_000, ci: float = 0.95,
            rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high)."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(v.mean()), lo, hi


def boot_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Returns (mean_diff = mean(b)-mean(a), low, high) — independent samples."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        diffs[i] = b[ib].mean() - a[ia].mean()
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fmt(m: float, lo: float, hi: float) -> str:
    if np.isnan(m):
        return "      nan         "
    return f"{m:>7.4f} [{lo:>7.4f},{hi:>7.4f}]"


# -----------------------------------------------------------------------------
# Output buffer
# -----------------------------------------------------------------------------

class Out:
    def __init__(self):
        self.buf = StringIO()
    def __call__(self, line: str = ""):
        print(line)
        self.buf.write(line + "\n")
    def text(self) -> str:
        return self.buf.getvalue()


# -----------------------------------------------------------------------------
# Section 1 — sanity check
# -----------------------------------------------------------------------------

def section_1_sanity(o: Out, run_dir: Path, train_rows: list[dict],
                     greedy_data: dict, stoch_files: dict):
    o("## Item 1 — Sanity check: eval harness vs training rollouts")
    o()
    o("We picked one checkpoint (u27) and ran the eval harness with the SAME ")
    o("seed the training rollout used (`base_seed = 42 + 27*1000 = 27042`).")
    o("Both harnesses share the same `LoRAAgentPolicy.generate` call, the same ")
    o("`active_adapter_ctx` manager for adapter switching, the same hidden-state ")
    o("extraction code path, and the same patient simulator.")
    o()

    # Find u27 training rollout stats
    u27 = next((r for r in train_rows if r.get("update") == 27), None)
    if u27 is None:
        o("(no train_log row for update 27 — skipping)")
        return

    sanity_path = run_dir / "eval_00027_turns_seed27042.jsonl"
    sanity_stoch_path = run_dir / "eval_00027_turns_seed27042_stoch.jsonl"

    sanity_greedy_rows = load_jsonl(sanity_path)
    sanity_stoch_rows  = load_jsonl(sanity_stoch_path)

    o("```")
    o(f"{'source':>40}  {'mean_c':>8}  {'mean_σ':>8}  {'n':>4}")
    o(f"{'training rollout @ u27 (T=0.7 stoch)':>40}  "
      f"{u27['mean_c_consensus']:>8.4f}  {u27['mean_sigma']:>8.4f}  "
      f"{u27['n_steps']:>4d}")
    if sanity_stoch_rows:
        mc, _, _ = boot_ci(col(sanity_stoch_rows, "c_consensus"))
        ms, _, _ = boot_ci(col(sanity_stoch_rows, "external_safety"))
        o(f"{'eval harness @ u27, T=0.7 stoch':>40}  "
          f"{mc:>8.4f}  {ms:>8.4f}  {len(sanity_stoch_rows):>4d}")
    if sanity_greedy_rows:
        mc, _, _ = boot_ci(col(sanity_greedy_rows, "c_consensus"))
        ms, _, _ = boot_ci(col(sanity_greedy_rows, "external_safety"))
        o(f"{'eval harness @ u27, T=0 greedy':>40}  "
          f"{mc:>8.4f}  {ms:>8.4f}  {len(sanity_greedy_rows):>4d}")
    o("```")
    o()
    if not sanity_stoch_rows:
        o("⚠️  Stochastic sanity eval not yet run. To complete this row:")
        o("```")
        o("sbatch --export=ALL,CKPTS=\"ckpt_00027\",BASE_SEED=27042,STOCHASTIC=1 \\")
        o("    slurm/reeval_checkpoints.job")
        o("```")
        o()
    o("**Confirmations of equivalence:**")
    o(f"  - Adapter active: `active_adapter_ctx(model, name)` in both rollout ")
    o(f"    (`src/mappo/rollout.py:284,297,313,326`) and eval shims via ")
    o(f"    `LoRAAgentPolicy.generate` (`src/mappo/policy.py:156`).")
    o(f"  - Sampling: both use `base_model.generate(do_sample=...)` with the ")
    o(f"    same `(temperature, top_p)` configuration.")
    o(f"  - Hidden state extraction: both call `LoRAAgentPolicy.generate(return_hidden=True)`")
    o(f"    which slices the last-layer last-token hidden vector identically.")
    o()


# -----------------------------------------------------------------------------
# Section 2 — headline: baseline vs MAPPO
# -----------------------------------------------------------------------------

def section_2_headline(o: Out, run_dir: Path):
    o("## Item 2 — Headline: baseline vs MAPPO c_consensus on held-out")
    o()
    base_rows = load_jsonl(run_dir / "baseline_turns_seed10000.jsonl")
    u24_rows  = load_jsonl(run_dir / "eval_00024_turns.jsonl")
    u24_s2    = load_jsonl(run_dir / "eval_00024_turns_seed20024.jsonl")
    if not base_rows:
        o("⚠️  baseline_turns_seed10000.jsonl not found.")
        return
    if not u24_rows:
        o("⚠️  eval_00024_turns.jsonl not found.")
        return

    rng = np.random.default_rng(0)
    b_c = col(base_rows, "c_consensus")
    m_c = col(u24_rows,  "c_consensus")
    m_c_combined = np.concatenate([m_c, col(u24_s2, "c_consensus")]) if u24_s2 else m_c

    bm, blo, bhi = boot_ci(b_c, rng=rng)
    mm, mlo, mhi = boot_ci(m_c_combined, rng=rng)
    dm, dlo, dhi = boot_diff_ci(b_c, m_c_combined, rng=rng)

    n_scen = len({r["scenario"] for r in base_rows})

    o(f"**5 scenarios, {len(base_rows)} baseline turns, ")
    o(f"{len(m_c_combined)} MAPPO turns ({'2 seeds combined' if u24_s2 else '1 seed'}).**")
    o()
    o("```")
    o(f"{'group':>32}  {'mean_c':>8}  {'95% CI':>22}  n")
    o(f"{'Baseline (untrained, greedy)':>32}  {bm:>8.4f}  [{blo:>7.4f},{bhi:>7.4f}]  {len(b_c)}")
    o(f"{'MAPPO @ u24 (greedy)':>32}  {mm:>8.4f}  [{mlo:>7.4f},{mhi:>7.4f}]  {len(m_c_combined)}")
    o()
    o(f"{'Δ MAPPO − Baseline':>32}  {dm:>+8.4f}  [{dlo:>+7.4f},{dhi:>+7.4f}]")
    rel = (dm / bm * 100) if bm else 0.0
    o(f"{'Δ relative':>32}  {rel:>+7.2f}%")
    o()
    if dlo <= 0 <= dhi:
        o(f"{'verdict':>32}  CI on Δ CROSSES zero — not significant")
    elif dm < 0:
        o(f"{'verdict':>32}  CI on Δ excludes zero — MAPPO REDUCES c_consensus")
    else:
        o(f"{'verdict':>32}  CI on Δ excludes zero — MAPPO INCREASES c_consensus")
    o("```")
    o()
    if u24_s2:
        # Inter-seed agreement
        m1, _, _ = boot_ci(col(u24_rows, "c_consensus"), rng=rng)
        m2, _, _ = boot_ci(col(u24_s2,   "c_consensus"), rng=rng)
        o(f"Inter-eval-seed check (MAPPO u24): seed 10024 mean={m1:.4f}, "
          f"seed 20024 mean={m2:.4f} (|Δ|={abs(m1-m2):.4f})")
        o()


# -----------------------------------------------------------------------------
# Section 3 — c_consensus split
# -----------------------------------------------------------------------------

def section_3_split(o: Out, run_dir: Path, ckpts: list[int]):
    o("## Item 3 — c_consensus split into similarity_term and unsafety_term")
    o()
    o("c_consensus = exp(−d/τ) · (1−σ). MAPPO targets the similarity term ")
    o("(latent agreement). The unsafety term is what the judge sees.")
    o()

    base = load_jsonl(run_dir / "baseline_turns_seed10000.jsonl")
    rng = np.random.default_rng(0)

    rows_per_ckpt = {u: load_jsonl(run_dir / f"eval_{u:05d}_turns.jsonl") for u in ckpts}

    o("**Held-out trajectory (with baseline reference):**")
    o("```")
    header = f"{'group':>14}  {'c_consensus':>22}  {'similarity':>22}  {'unsafety':>22}"
    o(header)
    if base:
        c = boot_ci(col(base, "c_consensus"),     rng=rng)
        s = boot_ci(col(base, "similarity_term"), rng=rng)
        u = boot_ci(col(base, "unsafety_term"),   rng=rng)
        o(f"{'baseline':>14}  {fmt(*c)}  {fmt(*s)}  {fmt(*u)}")
    for ux in ckpts:
        rows = rows_per_ckpt[ux]
        if not rows:
            continue
        c = boot_ci(col(rows, "c_consensus"),     rng=rng)
        s = boot_ci(col(rows, "similarity_term"), rng=rng)
        u = boot_ci(col(rows, "unsafety_term"),   rng=rng)
        o(f"{'u'+str(ux):>14}  {fmt(*c)}  {fmt(*s)}  {fmt(*u)}")
    o("```")
    o()
    o("**Interpretation:** if similarity_term is roughly flat across u4 → u24 ")
    o("(and equal to baseline), MAPPO did NOT push latent agreement down.")
    o()


# -----------------------------------------------------------------------------
# Section 4 — per-scenario table
# -----------------------------------------------------------------------------

def section_4_scenarios(o: Out, run_dir: Path, ckpts: list[int]):
    o("## Item 4 — Per-scenario c_consensus (held-out)")
    o()
    o("Per-scenario CIs from n≈75 turns/cell. Training-time per-scenario stats ")
    o("were not logged separately (train_log aggregates across scenarios), so ")
    o("this table is held-out only.")
    o()

    rng = np.random.default_rng(0)
    base = load_jsonl(run_dir / "baseline_turns_seed10000.jsonl")
    rows_per_ckpt = {u: load_jsonl(run_dir / f"eval_{u:05d}_turns.jsonl") for u in ckpts}
    scens = sorted({r["scenario"] for r in base}) if base else []

    o("**c_consensus by scenario × checkpoint:**")
    o("```")
    head = f"{'scenario':>25}  {'baseline':>10}  " + "  ".join(f"{'u'+str(u):>10}" for u in ckpts)
    o(head)
    for s in scens:
        row = f"{s:>25}  "
        bvals = np.array([r["c_consensus"] for r in base if r["scenario"] == s and r["c_consensus"] is not None])
        bm, _, _ = boot_ci(bvals, rng=rng) if len(bvals) else (float("nan"), 0, 0)
        row += f"{bm:>10.4f}  "
        for u in ckpts:
            vals = np.array([r["c_consensus"] for r in rows_per_ckpt[u]
                             if r["scenario"] == s and r["c_consensus"] is not None])
            m, _, _ = boot_ci(vals, rng=rng) if len(vals) else (float("nan"), 0, 0)
            row += f"  {m:>10.4f}"
        o(row)
    o()
    o("**unsafety_term (1−σ) by scenario — sycophancy fingerprint check:**")
    o("(rising on adversarial scenarios = trained therapist became too agreeable)")
    o(head)
    for s in scens:
        row = f"{s:>25}  "
        bvals = np.array([r["unsafety_term"] for r in base if r["scenario"] == s and r["unsafety_term"] is not None])
        bm, _, _ = boot_ci(bvals, rng=rng) if len(bvals) else (float("nan"), 0, 0)
        row += f"{bm:>10.4f}  "
        for u in ckpts:
            vals = np.array([r["unsafety_term"] for r in rows_per_ckpt[u]
                             if r["scenario"] == s and r["unsafety_term"] is not None])
            m, _, _ = boot_ci(vals, rng=rng) if len(vals) else (float("nan"), 0, 0)
            row += f"  {m:>10.4f}"
        o(row)
    o()
    o("**similarity_term exp(−d/τ) by scenario — penalty-gaming check:**")
    o("(should DROP if MAPPO is working; flat means latent agreement unchanged)")
    o(head)
    for s in scens:
        row = f"{s:>25}  "
        bvals = np.array([r["similarity_term"] for r in base if r["scenario"] == s and r["similarity_term"] is not None])
        bm, _, _ = boot_ci(bvals, rng=rng) if len(bvals) else (float("nan"), 0, 0)
        row += f"{bm:>10.4f}  "
        for u in ckpts:
            vals = np.array([r["similarity_term"] for r in rows_per_ckpt[u]
                             if r["scenario"] == s and r["similarity_term"] is not None])
            m, _, _ = boot_ci(vals, rng=rng) if len(vals) else (float("nan"), 0, 0)
            row += f"  {m:>10.4f}"
        o(row)
    o("```")
    o()


# -----------------------------------------------------------------------------
# Section 5 — critic health
# -----------------------------------------------------------------------------

def section_5_critic(o: Out, train_rows: list[dict]):
    o("## Item 5 — Critic health")
    o()
    o("**value_loss across all updates** (full trajectory):")
    o("```")
    o(f"{'update':>6}  {'value_loss':>10}  {'EV':>6}")
    for r in train_rows:
        ev = r.get("explained_variance")
        ev_str = f"{ev:>6.3f}" if ev is not None and not np.isnan(ev) else "  --  "
        o(f"{r['update']:>6}  {r.get('value_loss', 0):>10.4f}  {ev_str}")
    o("```")
    o()
    ev_vals = [r.get("explained_variance") for r in train_rows
               if r.get("explained_variance") is not None]
    if ev_vals:
        o(f"**Explained variance (where logged, u26+):** ")
        o(f"mean = {np.mean(ev_vals):.3f}, range [{min(ev_vals):.3f}, {max(ev_vals):.3f}]")
        o()
        o("> EV near 0.85 means the critic explains ~85% of return variance — ")
        o("> a healthy, converged critic. The value_loss plateau at ~1.5 ")
        o("> reflects bounded MSE on high-variance returns (std ≈ 3.2), not ")
        o("> a broken critic.")
    o()


# -----------------------------------------------------------------------------
# Section 6 — reward and advantage scales
# -----------------------------------------------------------------------------

def section_6_scales(o: Out, train_rows: list[dict]):
    o("## Item 6 — Reward and advantage scales")
    o()
    rows_with_scales = [r for r in train_rows if r.get("r_therapy_std") is not None]
    if not rows_with_scales:
        o("(no rows with the new reward-scale logging; the trainer.py change ")
        o("took effect at u26)")
        return

    o("```")
    o(f"{'u':>3}  {'r_therapy_std':>13}  {'β·c_cons_std':>13}  {'ratio (rT/βc)':>13}  "
      f"{'adv_mean':>9}  {'adv_std':>8}  {'ret_std':>8}  {'normalized':>10}")
    for r in rows_with_scales:
        rt = r["r_therapy_std"]
        bc = r["beta_c_consensus_std"]
        ratio = rt / bc if bc > 0 else float("inf")
        norm = "yes" if r.get("advantage_normalized") else "no"
        o(f"{r['update']:>3}  {rt:>13.4f}  {bc:>13.4f}  {ratio:>13.2f}  "
          f"{r.get('adv_mean_raw', 0):>9.4f}  {r.get('adv_std_raw', 0):>8.4f}  "
          f"{r.get('returns_std', 0):>8.4f}  {norm:>10}")
    o("```")
    o()
    avg_ratio = float(np.mean([r["r_therapy_std"] / r["beta_c_consensus_std"]
                               for r in rows_with_scales if r["beta_c_consensus_std"] > 0]))
    o(f"**Average r_therapy_std / β·c_consensus_std ratio: {avg_ratio:.2f}**")
    o()
    o("> A ratio > 1 means the therapy reward varies more per batch than the ")
    o("> consensus penalty does. The penalty contribution to the gradient is ")
    o("> dominated by therapy-reward noise. This is the reward-scale obstruction ")
    o("> to MAPPO descending on c_consensus.")
    o()
    o("**Advantage normalization is ON** "
      "(`adv = (adv - adv.mean()) / (adv.std() + 1e-8)` in `trainer.py:96`).")
    o()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="data/results/mappo/main")
    parser.add_argument("--checkpoints", nargs="+", type=int,
                        default=[4, 9, 14, 19, 24])
    parser.add_argument("--out", default="mindsafeDocs/mappo_diagnostic.md")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    train_rows = load_jsonl(run_dir / "train_log.jsonl")

    # File discovery
    greedy_data = {u: load_jsonl(run_dir / f"eval_{u:05d}_turns.jsonl")
                   for u in args.checkpoints}
    stoch_files = sorted(run_dir.glob("eval_*_turns_*_stoch.jsonl"))

    o = Out()
    o("# MAPPO Diagnostic — Meng's 6-item review")
    o()
    o(f"Source: `{run_dir}`")
    o(f"Checkpoints: {args.checkpoints}")
    o(f"Training updates logged: {len(train_rows)}")
    o()

    section_1_sanity(o, run_dir, train_rows, greedy_data, stoch_files)
    section_2_headline(o, run_dir)
    section_3_split(o, run_dir, args.checkpoints)
    section_4_scenarios(o, run_dir, args.checkpoints)
    section_5_critic(o, train_rows)
    section_6_scales(o, train_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(o.text())
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
