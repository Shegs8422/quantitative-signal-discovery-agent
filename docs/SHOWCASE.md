# Showcase Brief — Quantitative Signal Discovery Agent (Fork)

One-line pitch: **an autonomous multi-agent LLM that reads equity/ETF panels, invents statistically testable alpha hypotheses, codes them, backtests them — and (in this fork) refuses to fool you with inflated statistics, bearish beta, or honest-to-goodness costs.**

Base: NVIDIA's `quantitative-signal-discovery-agent` blueprint (Apache-2.0, forked; additive changes only).
Stack: NVIDIA NeMo Agent Toolkit + Nemotron 3.5 Lightning (hosted NIM), yfinance, pandas/scipy, Arize Phoenix.

## The 5-minute demo script

1. `uv sync --locked && .venv\Scripts\python.exe -m pip install "nvidia-nat[profiler]"`
2. `uv run python -m signal_discovery_workflow.download_data --universe bond_etfs`
3. `uv run phoenix serve`  (separate terminal → http://localhost:6006)
4. `uv run --env-file .env nat run --config_file configs/config-optimization-hac.yml --input "momentum signals"`
5. Point at the run's output JSON + saved signal in `src/signal_discovery_workflow/output/`
6. `uv run python -m signal_discovery_workflow.signal_report src/signal_discovery_workflow/output/signal_xxx.json --data-dir src/signal_discovery_workflow/data/bond_etfs`
7. `uv run python -m signal_discovery_workflow.portfolio_sim src/signal_discovery_workflow/output/signal_xxx.json --data-dir src/signal_discovery_workflow/data/bond_etfs --rebalance monthly`

Talking points, in order:
- **Loop architecture** — Signal → Code → Evaluation → advisor feedback; self-corrects operator arity/name errors.
- **Grounding** — composite formulas from a fixed, vetted 66-operator catalogue; deterministic operator inlining.
- **Hallucination tolerance** — ~1–2 non-catalogue names per iteration (e.g. `TS_Volatility`); they die at execution, the loop observes, retries; the fork also aliases near-misses.
- **Honest statistics** — the headline fork addition: HAC inference as an opt-in *acceptance gate*, cost drag as first-class output, Bonferroni trial registry, FX attribution for international panels.
- **The catch** (live example) — Iteration 1, naive p = 0.031 (upstream accepts) → **HAC p = 0.203 → gate rejects**. Iteration 2 accepted instead (HAC p = 0.0025).
- **The inconvenient autopsy** — every accepted signal fails deployability once costs, beta and regimes are priced in. The honest finding: quality structure ≠ quality portfolio.

## What the fork extends vs. upstream

| Area | Extension |
|---|---|
| Data | Nasdaq 100, Dow 30, country/sector/bond/global-bond/factor ETFs; custom `--tickers`; `$SIGNAL_DATA_DIR` switching; FX pairs + local-currency re-execution |
| Statistics | HAC everywhere; opt-in acceptance gate; bootstrap CIs; horizon ladder; autocorrelation; regime splits; market-neutral residualized IC |
| Trading realism | Turnover + 10 bps one-way cost drag; `cost_warning` in workflow results; portfolio construction with vol targeting |
| Integrity | Quoting-safe resume driver; trial registry; direction disclosure; provenance docs; 115/115 tests; ruff clean |
| Observability | `nvidia-nat[profiler]` → per-LLM spans + token counts in Phoenix |

## Numbers from this machine (2026-11-12)

| Metric | Value |
|---|---|
| Workflow executions | 10 |
| Accepted signals | 3 of 10 (bond_etfs, global_bond_etfs — momentum family) |
| HAC-gate rejections of naive-acceptable candidates | 1 |
| LLM spans captured | 6/6 per run (after profiler fix; costs still unpriced for this model) |
| Deepest finding | accepted signal = beta, not selection (residualized IC ≈ 0, p = 0.42) |
| Investability verdict | 0 of 3 accepted signals survive portfolio economics |

## Honest limitations (say these out loud)

Survivorship bias, no delisted streams, no slippage/impact/borrow desk, overlapping horizons, small panels (k=2 buckets), no FX hedge cost model, single-model (no ensemble), multiple testing only partially mitigated (Bonferroni + trial registry). More in `docs/INVESTMENT_ASSURANCE.md`.

## Suggested close

> "This is not an alpha machine. It is a machine for disqualifying bad ideas fast — with receipts. The value is the honesty gate: HAC inference, cost drag, regime splits, beta-neutral checks — running in minutes instead of weeks. Three signals passed the discovery gate; three failed inspection. That is exactly where 95% of LLM-generated signals should end up."
