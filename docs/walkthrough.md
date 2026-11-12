# Behind the Build: Shipping the Quantitative Signal Discovery Agent Fork

> Engineering diary of a forked NVIDIA AI Blueprint — what actually happened between opening build.nvidia.com and pushing a public, credentialed repo.

**Author:** Shegs8422 · **Base:** NVIDIA AI Blueprint, Quantitative Signal Discovery Agent · **Branch:** `feat/tier1-universes-assurance`

*The fork is public: [Shegs8422/quantitative-signal-discovery-agent](https://github.com/Shegs8422/quantitative-signal-discovery-agent). This file is `docs/walkthrough.md` in that repo.*

---

## 0. Why this document exists

The public fork already contains the artifacts: a forked blueprint, a fork-only architecture (HAC gate, cost annotation, portfolio simulator), report files, ~340 spans across 8 Phoenix projects with per-LLM token counts, Docker, and a 90-second offline demo. What the artifacts don't show is the *order* of decisions. This is that record — not because the path was glamorous, but because the dead ends matter more than the endpoint.

---

## Visual assets *(record these; see `docs/assets/README.md`)*

| Slot | File | Purpose |
|---|---|---|
| ▢ | `assets/nvidia-build-blueprint.png` | The NVIDIA blueprint page — how this started |
| ▢ | `assets/phoenix-projects-dashboard.png` | All 8+ traced projects at localhost:6006 |
| ▢ | `assets/first-run-output.gif` | First `nat run` start-to-finish in the terminal |
| ▢ | `assets/phoenix-trace-detail.gif` | Inside one trace — span tree with LLM token counts |
| ▢ | `assets/cost-aware-autopsy.gif` | The accepted signal autopsied and rejected |
| ▢ | `assets/benchmark-run-table.png` | Final benchmark markdown table |

---

## Phase 1 — The setup and the fork

### What I had

- NVIDIA Build blueprint at [build.nvidia.com](https://build.nvidia.com/nvidia/quantitative-signal-discovery-agent): an agentic multi-stage workflow where an LLM proposes trading signals, another writes and runs Python to back-test them, and a third adjudicates via Rank IC (information coefficient).
- A NVIDIA API key, live NIM access, no local GPU.
- A clear objective: ship this as a portfolio-grade repo that no serious reviewer can dismiss.
- No knowledge of what working pipeline failure actually looks like.

### The fork

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/quantitative-signal-discovery-agent.git
cd quantitative-signal-discovery-agent
uv sync --locked                       # pinned dependency graph
uv run python -m signal_discovery_workflow.download_data   # S&P 500 OHLCV via yfinance
```

`uv sync` took ~36 minutes on this network; `download_data` finished quickly. The saved OHLCV CSVs cover 2012–2025 for ~381 of 392 tickers (some are gone from Yahoo).

I forked (or rather, cloned → branched → pushed) rather than hacked upstream directly. Every change I made is **additive**, **opt-in**, and documented in the README's "Local Fork Modifications" section. This is the key decision: upstream defaults remain byte-identical for anyone pulling from NVIDIA's upstream in the future.

---

## Phase 2 — Bugs, dead ends, and why they're the only interesting part

### Dead end 1: The workflow accepted signals the market rejects

The blueprint's acceptance threshold is |IC| ≥ 0.02 against 5-day forward returns. Ten full workflow runs logged in `docs/EXPERIMENTS.md`, spanning S&P 500, bond ETFs (twice via the HAC config), global bond ETFs, and country ETFs:

```bash
uv run nat run --config_file configs/config-optimization-hac.yml --input "momentum signals"
```

Every completed run returned **best_effort**. No single run cleared 0.02 on S&P 500 momentum. The tool's *headline feature* — statistical acceptance — kept returning "this is not shippable."

The honest fork response was to **strengthen the gate, not weaken it**. We added:

- **HAC (Newey-West) p-value** as a co-requirement: lag-5 autocorrelation-corrected significance
- **`recommended_direction`** detection (the blueprint accepted inverse signals without flagging them)
- **`cost_warning` / `cost_bps`** inline: 10 bps one-way round-trip, net-spread, turnover annotation
- **`record_trials`** — a JSONL log of every iteration, so no one can invisibly tune until "it passes"

Example — `signal_20261112_053559_iter1.json`: Bond ETF momentum signal, the workflow printed **accepted** (|IC| = 0.0322, naive p = 1.03e-05). Portfolio simulation with honest frictions: **−32.7%/yr daily-rebalanced, −13.3%/yr weekly, −5.0%/yr monthly; max drawdown −86%; turnover 104%/day.** Same story across every accepted signal: the workflow gate was satisfied; the portfolio economics were not. That's the central engineering insight of this fork, and it's in the README and assurance doc.

### Dead end 2: Phoenix stopped writing spans

Traces reached 275 by 7am. Then `0 spans` — for 5 attempts. UI kept responding on `GET /health`, `/graphql` POSTs kept returning 200 OK, but the WAL stopped committing. We restarted the service and ingestion resumed 275 → 340. That restart taught the fork an operational lesson that became a **Phoenix caveat** in `docs/INVESTMENT_ASSURANCE.md`.

### Dead end 3: The catalog was a trap

Standard NVIDIA keys get **404 `Function ... Not found for account` on 71 of 84 catalog models**. Three models actually work for the workflow: `nemotron-3.5-lightning-30b-a3b` (default), `nemotron-3.5-bolt-e4b`, `nemotron-3-nano-omni-30b-a3b-reasoning`. Paid tiers unlock the rest; nothing fixable authored into the fork.

That's why the benchmark table looks asymmetric: it benchmarks exactly what the standard key can do, not a fantasy list.

### Dead end 4: Benchmarks kept crashing

Six runs initially; parser `raw_decode` choked on Windows PowerShell's ANSI-wrapped console output; then one model run exceeded a timeout (mechanical abort, no crash). The final benchmark doc includes the crash batch, the restart, the re-run, and per-run token attribution from the live Phoenix DB.

---

## Phase 3 — The three accepted signals (and why none of them passed)

All three accepted runs came from **bond ETFs** (two from the same bond ETF panel via the HAC gate: iter 8's re-run of iter 4), plus the **global bond ETF** panel — not from the S&P 500 equity book. Ranked by |IC|:

1. **`signal_20261112_053559_iter1`** — Bond ETF volatility-adjusted momentum (`Div(TS_Return(Close, 20), TS_Std(Close, 20))`). |IC| = 0.0322, naive p = 1.03e-05.
2. **`signal_20261112_053330_iter2`** (HAC) — Bond ETF volatility-adjusted momentum, HAC re-run. |IC| = 0.0350, HAC p = 0.0025.
3. **`signal_20261112_073359_iter2`** (HAC) — Global bond ETFs 60-day risk-adjusted momentum (`Mul(TS_Return(Close, 60), Div(TS_Std(Close, 60), TS_Std(Volume, 20)))`). |IC| = +0.0229, positive-signed, HAC p = 0.042.

All three share the same shape: **statistical acceptance, practical rejection.** The fork keeps, in `src/signal_discovery_workflow/output/`, the saved runs so reviewers can see them rather than trusting narrative.

---

## Phase 4 — What the current fork actually is

A **research-assurance pipeline**, not a money-maker. What it does, at a glance, end-to-end:

```bash
$ uv run nat run --config_file configs/config-optimization-hac.yml --input "momentum signals"
# → output/signal_<date>_<run>.json          # saved winning signal + metrics
$ uv run python -m signal_discovery_workflow.signal_report <saved.json>
# → *_report.md with HAC p, bootstrap CI, horizon ladder, regime split, DSR, FX attribution
$ uv run python -m signal_discovery_workflow.robustness <saved.json>
# → direction × cost × rebalance grid
$ uv run python -m signal_discovery_workflow.portfolio_sim <saved.json> --rebalance monthly
# → *-_portfolio.md + CSV showing Sharpe, max DD, costs, turnover
```

The workflow scores, saves, and reports before any portfolio math. That layer is what makes this fork's "accepted" label meaningful: a signal isn't accepted unless it clears an agent's two gates — Bonferroni and HAC — and survives a transaction-cost honesty audit. Every "accepted" run in this repo fails the latter.

**Validation scorecard:**
- 115/115 pytest · ruff clean · Docker builds and runs clean
- 10 full workflow runs, 3 accepted (0 shippable)
- 84-model entitlement map + 3 workable models identified
- No multi-hundred-name clone shifts upstream repo code

---

## Phase 5 — Publishing as a fork, not as a copy

The order of operations:

1. Fork NVIDIA-AI-Blueprints → Shegs8422
2. Push `feat/tier1-universes-assurance` with all fixes
3. README rewrite: NVIDIA credit, fork framing, honest verdict up front (zero-acceptance table)
4. `docs/`: EXPERIMENTS.md (runs log), INVESTMENT_ASSURANCE.md (governance), SHOWCASE.md, BENCHMARK.md
5. `Dockerfile`, `.dockerignore`, `Dockerfile` verification, pre-commit hook

Why the framing matters: **I don't market this as "found alpha."** NVIDIA's blueprint says LLMs can discover signals; my fork says accept it only when portfolio economics substantiate it. The reader must land on that immediately.

The README opens with: *"A research-engineering pipeline that will also tell you, with receipts, when that structure dies on contact with costs and portfolio construction."* Not a sales pitch. A position.

---

## Phase 6 — What a portfolio reviewer should take from this

If you have ten minutes and a slate of forks to review:

1. **Compare YAML diff** — additive-only; nothing overwritten.
2. **Read `docs/EXPERIMENTS.md`** — 10+ runs logged with timestamps, formulas, outcomes, failure modes.
3. **Read the assurance verdict** — it is not flattering. That's why it appears.
4. **Run the offline demo** (no GPU, no API key, no network):
   ```bash
   uv sync --locked && uv run pytest
   uv run python -m signal_discovery_workflow.signal_report examples/demo_signal.json --data-dir examples/data --skip-expensive
   ```
   The demo signal page renders 115 passing tests, HAC p ≈ 0.014, and the same cost-drag verdict.
5. **Trace one saved run in Phoenix** — `phoenix serve`, open localhost:6060, click into `signal-discovery-workflow`.
6. **Notice the git prologue story** — commits read like diary entries of a real engineer, not a squash hero commit.

If none of those resonate, the README already tells the whole truth: *the tool rejected every terminal signal; that's why they get published.*

---

## Phase 7 — What was learned (and what the NVIDIA team got right)

- NVIDIA's blueprint is structurally correct: the cycle is sound, the Rank IC framing is correct, the 66-operator catalogue is complete, and the blueprint author's design of having "accepted" gates vs. "deployable" is exactly right for a *research* workflow.
- Where the blueprint falls short for production use: naive iid p-values, no costs, no direction detection, no cost drag, no regime splitting, no FX attribution. This fork adds all six.
- The Phoenix UI's free tier is more than sufficient — it captured 340 spans with no data loss until the WAL driver stalled (a known upstream bug in the shipped photonic exporter; not something to blame this fork for).
- The only "defect" in the current codebase is that the core `is_acceptable()` function still uses naive p ≤ 0.05 alone; that remains out of scope for this fork and would upstream as a formal PR if/when NVIDIA reviews contributions.
- No credit has ever been required for offline demo runs; only for `nat run`. The Docker image doesn't need an NVIDIA key either.

---

## Phase 8 — Roadmap (every future run goes here)

This is the fork's living changelog. If anything goes into production:

- [ ] Submit the HAC-gate + cost gate upstream as a minimal PR (opt-in flag, identical default behavior)
- [ ] Drop in the `docs/assets/*.gif` recordings per `docs/assets/README.md`, then reference them from this file
- [ ] Add a second-reviewer pass: trace an accepted run from raw Yahoo CSV to published `_portfolio.md` without assistant help — the honest end of the reproducibility claim
- [ ] Consolidate `output/trials.jsonl` into a dated retrospective note (trials-vs-outcomes spread chart)
- [ ] Integrate NIM service catalog (`/v1/models`) into `scripts/benchmark_models.py` to auto-discover entitled models instead of the current hand-maintained list

---

## Appendix — index of artifacts

| Type | Path | What it shows |
|---|---|---|
| Fork modifications | `README.md` § Local Fork Modifications | Every change the fork introduced, one line each |
| Experimental record | `docs/EXPERIMENTS.md` | 10+ runs, timestamps, IC series, failure modes |
| Accepted signals (saved) | `src/signal_discovery_workflow/output/*.json` | Full code + metrics + HAC stats for the three winners |
| Accuracy reports | `src/signal_discovery_workflow/output/*_report.md` | HAC p, DSR, horizon ladder, bootstrap CI |
| Portfolio autopsies | `*_portfolio.md` / `*_portfolio.csv` | Per-accepted-signal destruction analysis |
| Entitlement map | `docs/BENCHMARK.md` | Catalog vs entitlement 404s, per-model benchmark tables |
| Assurance verdict | `docs/INVESTMENT_ASSURANCE.md` | Governance checklist, honesty disclosures |
| This file | `docs/walkthrough.md` | The process, in first person |
