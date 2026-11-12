# Experiment Log — Signal Discovery Fork (2026-11-12)

Branch: `feat/tier1-universes-assurance` | Base: NVIDIA `NVIDIA-AI-Blueprints/quantitative-signal-discovery-agent` @ `main` (9a89d24)
Environment: Windows 11, Python 3.12.10, NAT 1.6.0, `nemotron-3.5-lightning-30b-a3b` via `integrate.api.nvidia.com`, all inference hosted (CPU-only local eval).
Data: S&P 500 → 381/392 tickers (Yahoo gaps: CTRA, HOLX, BK, FI, EA, AVB, EQR, CMA, IPG, K, MMC); bond_etfs → 12/12; global_bond_etfs → 28/28; country_etfs → 20/20; FX → 15/15 pairs; 3,519 bars each (2012–2025).

## Results matrix

| Run | Universe | Input | Seed | Status | mean IC | |IC| | p-value | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sp500 | momentum signals | — | best_effort | −0.0119 | 0.0119 | 1.6e-06 | baseline; saved iter1 |
| 2 | sp500 | resume momentum | broken string* | best_effort | — | 0.0053 | 0.031 | invalid experiment |
| 3 | sp500 | resume momentum | broken string* | best_effort | −0.0101 | 0.0101 | 6.1e-04 | invalid experiment |
| 4 | bond_etfs | momentum signals | — | **accepted** ✅ | **−0.0322** | **0.0322** | **1.0e-05** | iter 1; saved |
| 5 | sp500 | mean reversion signals | — | best_effort | −0.0167 | 0.0167 | 2.6e-10 | t = −6.3 |
| 6 | sp500 | volatility signals | — | best_effort | −0.0048 | 0.0048 | 0.016 | weak |
| 7 | sp500 | mean reversion signals | fixed resume driver | best_effort | −0.0109 | 0.0109 | 1.6e-04 | seed verified; didn't beat seed |
| 8 | bond_etfs | momentum, `config-optimization-hac.yml` | — | **accepted** ✅ | **−0.0350** | 0.0350 | naive 1.1e-04 · **HAC p = 0.0025** | iter 2; HAC gate live |
| 9 | global_bond_etfs | momentum, HAC config | — | **accepted** ✅ | **+0.0229** | 0.0229 | naive 2.3e-04 · **HAC p = 0.042** | iter 2, positive direction |
| 10 | country_etfs | momentum, HAC config | — | best_effort | −0.0166 | 0.0166 | naive 1.1e-04 · HAC p = 0.036 | significance w/o magnitude |

`*` Windows PowerShell does not safely pass native args with embedded quotes; the JSON `--input` arrived as a raw blob and `seed_feedback` was never applied. Runs 7/8/9/10 used a Python `subprocess` argv list.

## What runs 8 and 9 proved (HAC gate live)

- Run 8, iteration 1: naive p = 0.031 (**would pass upstream's rule**) but HAC p = 0.203 → **rejected by the HAC gate**. The exact false-positive class this gate exists to kill. Iteration 2 accepted (HAC p = 0.0025).
- Run 9: accepted at HAC p = 0.042 — would fail stricter (0.01) gates.
- Result JSON now carries `hac_t_stat`, `hac_p_value`, `avg_turnover`, `net_spread`, `cost_warning`, `recommended_direction`, `direction_note`.

## FX attribution (offline validation, `--local-currency`)

Same saved signal re-executed on local-currency panel (20 country ETFs, 15 FX pairs):
- USD-denominated: mean IC −0.0073 (HAC p = 0.421)
- Local-currency: mean IC −0.0183 (HAC p = 0.038)
FX conversion materially changes the research read; USD IC is noise while local IC clears 5%. (Plumbing validation, not a research claim.)

## Deep validation of accepted signals

### Bond ETF momentum (runs 4/8), report tooling
- Horizon ladder: 1d −0.019 / 5d −0.032 / 10d −0.036 / 21d −0.021 (p=0.36) / 63d −0.009 — edge dies by ~2–3 weeks
- Autocorrelation (lag-1): 0.90 — persistence, not predictive width
- Regime: 2018 −0.058, 2019 −0.168, 2022 −0.116, sign-flips 2016 +0.034 / 2024 +0.061 — unstable
- Market-beta-neutral IC: −0.0099 (p = 0.42) — edge is beta, not selection alpha
- Portfolio sim: −32.7%/yr daily / −13.3%/yr weekly / −5.0%/yr monthly; 104%/39%/14% one-way turnover; inverse-direction net spread ≈ zero after costs

### Global bond ETF 60-day signal (run 9)
- Bootstrap CI +0.0005…+0.046 (barely excludes 0); decile spread t = −0.04 (p = 0.97) — no tradable cross-section
- Sub-period decay +0.033 → +0.012; Bonferroni (6 trials) fails; portfolio sim: −2.34%/yr, Sharpe −0.25, max DD −36%

### S&P 500 mean reversion (run 5)
- HAC t = −3.69, p = 2.3e-04; L/S decile spread −0.0023 gross, −0.0026 net — inverse edge, cost-underwater as coded

## Observability (Phoenix + profiler)

`nvidia-nat-profiler` installed on top of `uv sync --locked` restores LLM spans: 6/6 LLM calls with token counts (e.g. 1751/387); `<workflow>` CHAIN spans accumulate tokens. `span_costs` stays empty — Phoenix has no price-table entry for `nvidia/nemotron-3.5-lightning-30b-a3b`. Note: `uv sync --locked` removes the profiler (it's absent from the lock); reinstall after every sync.

## Supporting data downloads

`data/sp500/` (381/392), `data/bond_etfs/` (12/12), `data/global_bond_etfs/` (28/28), `data/country_etfs/` (20/20), `data/fx/` (15/15 FX pairs), all 3,519 bars (2012–2025).

## Statistical honesty checklist (for showcase)
- [x] HAC-adjusted inference everywhere meaningful + opt-in acceptance gate (`config-optimization-hac.yml`)
- [x] Transaction-cost drag and turnover priced into every artifact
- [x] Sub-period stability, regime splits, horizon ladder, autocorrelation, residualized IC, block bootstrap
- [x] Multiple-testing registry (`output/trials.jsonl`) with Bonferroni annotation — 6+ report runs + workflow runs (`record_trials`)
- [ ] Survivorship bias (current constituents; delisted returns absent)
- [ ] Slippage / impact / borrow-facility economics unmodeled; FX hedge cost unmodeled
- [ ] Annualized figures overlap-inflated — reference only
- [ ] Bond panel: 12 names → k=2 buckets (illustrative only)
- [ ] Trial inflation by dev reruns — registry deliberately conservative

## Failure modes observed
1. **Operator hallucination** (most common): `TS_Volatility`, `EWM_Std`, `TS_Kurtosis`, `TS_MAD`, `TS_VolumeWeight`, `TS_StdDev` — none exist in `calculator.json`. Mitigation shipped: alias table in `llm_utils.OPERATOR_SYNONYMS`.
2. **|IC| sign blindness** — inverse signals accepted with no executor guidance → fixed via `recommended_direction`/`direction_note`.
3. **Shell-quoting of JSON `--input` on Windows PowerShell** — silently corrupted resume runs (runs 2–3) → fixed, runs 7+ verified via Python subprocess driver.
4. **Phoenix CHAIN-only tracing without profiler extra** — fixed, see above.

## Takeaway
The agent finds statistically significant cross-sectional structure cheaply, reliably, and repeatably — and that structure does not survive honest costs, beta adjustment, regime scrutiny, and portfolio construction, on any of the four universes tested. The tool's honest job description: **disqualify candidate alpha fast, with receipts** — not manufacture alpha.
