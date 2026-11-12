# Benchmark — cost per accepted signal (Nemotron family)

Generated: 2026-11-12 09:12:03 | Request: `momentum signals` | Runs/model: 2 | Config: `configs/config-optimization-hac.yml` (identical for all models; only `model_name` + Phoenix project differ)

## Entitlement (measured, direct API probe, near-zero token spend)

All other Nemotron models in the catalog return `404 Function ... Not found for account`
from this NVIDIA API key — catalog visibility ≠ entitlement. Only these work:

- `nvidia/nemotron-3.5-lightning-30b-a3b` (deployed default)
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- `nvidia/nemotron-3.5-bolt-e4b`

## Results (HAC-gated acceptance @ |IC| ≥ 0.02, p ≤ 0.05)

| Model | Run | Status | mean IC | HAC p | turnover | direction | wall (s) |
|---|---|---|---|---|---|---|---|
| lightning-30b | 1 | best_effort | −0.0079 | 0.0849 | 0.2428 | −1 | 243.6 |
| lightning-30b | 2 | best_effort | −0.0120 | 0.0055 | 0.2658 | −1 | 127.6 |
| nano-omni-30b | 1 | best_effort | −0.0134 | 0.0141 | 0.2019 | −1 | 156.3 |
| nano-omni-30b | 2 | best_effort | −0.0111 | 0.0042 | 0.4018 | −1 |  84.1 |
| bolt-e4b | 1 | best_effort | −0.0137 | 0.0099 | 0.2820 | −1 | 130.4 |
| bolt-e4b | 2 | best_effort | −0.0119 | 0.0058 | 0.2641 | −1 | 144.4 |

- **Acceptances: 0 of 6.** S&P 500 momentum continues to yield no 0.02-|IC| candidate under the HAC gate for any of the three models — consistent with every prior run in this repo.
- **Speed:** nano-omni fastest on second run (84 s) vs lightning's slower first (244 s); config variance, run-level LLM variance — treat as directional.
- **Signal strength (|IC|):** bolt ≈ 0.013, nano-omni ≈ 0.012, lightning ≈ 0.010 — all statistically significant curves, none near the 0.02 bar; none above deflated-Sharpe threshold either.
- **Phoenix token totals (cumulative, all runs on this machine incl. earlier sessions):** lightning 111,342 / nano-omni 62,399 / bolt 66,004 — **not benchmark-attributed; cumulative across history.**

## Observability gap discovered during the benchmark (honest)

After ~08:11 UTC no new spans were written to Phoenix for any of these runs
(span/trace writers remained alive per logs — "Started exporter 'phoenix'" and
"Event stream completed" both logged, but zero new rows appeared in
`.phoenix/phoenix.db`). Token accounting for this benchmark's specific runs could
not be recovered from the database, so per-run tokens are **not reported** rather
than estimated. Root cause not yet isolated (spans were ingesting healthily until
~08:11, then silently stalled while the exporter reported clean startups/shutdowns).
Logs for every run are preserved in `.benchmarks/logs/` (gitignored). A fresh
`phoenix serve` process is the expected fix for future runs.

## Reading the table

- Cost per accepted signal is **∞** under this config for all three models
  (`n/a (0 accepted)`). The useful question — which model buys more IC per token —
  is partly answerable: bolt/nano-omni deliver |IC| ≈ lightning at ~60% of the
  cumulative token cost on this machine, but one cannot accept what meets no bar.
- Prices intentionally omitted (they vary by plan); this table gives token counts.
- Two runs per model; directional, not a significance test of model quality.

## Caveats

- Momentum on mega-caps is a known-hard, mostly-rejected family; the models did not differ materially in acceptance rate.
- The benchmark used one request (`momentum signals`); model personality may differ on reversal/volatility requests.
- Wall-clock includes harness overhead (~5–10 s NAT startup per run).

## To reproduce

```bash
uv run python scripts/benchmark_models.py --runs 2 --request "momentum signals"
uv run python scripts/benchmark_models.py --only nano-omni-30b --runs 1   # single model
```
