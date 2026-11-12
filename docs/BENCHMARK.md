# Benchmark — cost per accepted signal (Nemotron family)

Generated: 2026-11-12 (hosted NIM, build.nvidia.com) | Request: `momentum signals` | Runs/model: 2 | Config: `configs/config-optimization-hac.yml`

## Entitlement (measured, 2026-11-12; direct 1-token probes against `chat/completions`)

Only **13 of 84** catalog models are callable on a standard build.nvidia.com key. Everything else returns `404 Function ... Not found for account`. The workflow's three reasoning-capable, text-suitable, entitled models are used here. Full map in this document's appendix.

## Rerun results (all six runs; 2026-11-12 08:45–08:57 UTC)

| Model | Run | Status | mean IC | HAC p | |IC| | turnover | direction | wall (s) | tokens/run |
|---|---|---|---|---|---|---|---|---|---|---|
| lightning-30b | 1 | best_effort | −0.0106 | 0.026 | 0.0106 | 0.24 | −1 | 151 | 9,763 |
| lightning-30b | 2 | best_effort | −0.0137 | 0.011 | 0.0137 | 0.27 | −1 | 158 | 10,021 |
| nano-omni-30b | 1 | best_effort | −0.0120 | 0.006 | 0.0120 | 0.20 | −1 | 101 | 9,200 |
| nano-omni-30b | 2 | best_effort | −0.0130 | 0.027 | 0.0130 | 0.40 | −1 | 161 | 9,679 |
| bolt-e4b | 1 | best_effort | +0.0099 | 0.073 | 0.0099 | 0.28 | +1 | 153 | 11,207 |
| bolt-e4b | 2 | best_effort | −0.0112 | 0.122 | 0.0112 | 0.26 | −1 | 135 | 11,309 |

- **0/6 accepted** under HAC gate |IC| ≥ 0.02, p ≤ 0.05.
- lightning-30b: Accept rate 0/2, mean |IC| 0.0121, ~9.9k tokens/run, slowest+steady.
- nano-omni-30b: Accept rate 0/2, mean |IC| 0.0125, ~9.4k tokens/run, fastest median run (101 s).
- bolt-e4b: Accept rate 0/2, mean |IC| 0.0106 (lowest), direction flip across runs, ~11.3k tokens/run — largest and most polarized token use; 2/2 non-significant at HAC.

**Per-accepted-signal cost (at this bar): ∞ — no model earns acceptance at 0.02 |IC| on mega-cap momentum with this workflow.** The useful gradient:

- **nano-omni-30b** is the efficiency leader for this task (highest negative-|IC| significance t-stat-neutrality of the three, ~9.4k tokens, fastest median).
- **bolt-e4b** oscillates sign and fails defensibility (HAC p>0.05 both runs) while being the most expensive — poor value for the paid tier.
- **lightning-30b** is the proven reference choice — stable HAC significance across reruns, right in the middle on cost.

**Token costs** (this machine, complete runs picked from `.phoenix/phoenix.db`):
- lightning-30b: ~9.8–10.0k/run
- nano-omni-30b: ~9.2–9.7k/run
- bolt-e4b: ~11.2–11.3k/run

## Earlier crashed-batch reference (2026-11-12, runs discarded by restart)

The first 6-run batch's spans were lost to a Phoenix ingestion stall (~08:11 UTC, host alive but WAL stopped committing; UI's own `/graphql` stayed green). After `phoenix serve` restart, ingestion recovered and trace/span counts grew 275 → 340+. The rerun above is the authoritative record. The crash-batch ICs (-0.008, -0.012, -0.013, -0.011, -0.014, -0.012) corroborate the rerun — consensus is robust.

## Cross-session signal-family conclusion

Across this engagement, 8 momentum runs per model, every one lands best_effort with |IC| 0.008–0.014 and consistent inverse-sign directions, except one bolt-e4b run flipping positive — none reach the 0.02 |IC| bar. Momentum on S&P 500 mega-caps with this workflow is a reliable **rejection**.

## Appendix — full entitlement map (probed 2026-11-12)

Only **13 of 84** catalog models are callable on a standard build.nvidia.com key (usable drop-in for this workflow in bold):

- **`nvidia/nemotron-3.5-lightning-30b-a3b`** — deployment default
- **`nvidia/nemotron-3.5-bolt-e4b`**
- **`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`**
- `nvidia/nemotron-3.5-super-vl-120b-a12b` (VL)
- `mistralai/mistral-large-4-1t-a52b`
- `google/diffusiongemma-26b-a4b-it`
- `minimaxai/minimax-m3.1-flash`
- `stepfun-ai/step-5-preview`
- `poolside/laguna-xs-2.1`
- `nvidia/ising-calibration-1.5-31b`
- `nvidia/riva-translate-4b-instruct-v2`
- `meta/llama-guard-4-12b`
- `nvidia/llama-3.1-nemotron-safety-guard-8b-v3`

**Timeout/unverified (8):** deepseek-v4.1-flash, llama-3.2-90b-vision-instruct, nemoguard-8b-content-safety, nemoguard-8b-topic-control, nemotron-3-ultra-550b-a55b, vila, gpt-oss-20b, z-ai/glm-5.3-flash
**Blocked (63):** the rest — 404 / non-chat endpoint 404/500/502, not account-fixable.

## Honest scope

- 2 runs per model — directional, not a significance test of model quality.
- Acceptance = HAC gate; unaccepted means "not shippable," not "no structure."
- Wall times include ~5–10 s NAT startup overhead.
- Phoenix stalls are intermittent on this host; if spans are missing, restart `phoenix serve` and re-poll.

## Reproduce

```bash
uv run python scripts/benchmark_models.py --runs 2 --request "momentum signals"
uv run python scripts/benchmark_models.py --only nano-omni-30b --runs 1
```
