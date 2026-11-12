# Quantitative Signal Discovery Agent developer example

An end-to-end signal discovery workflow for quantitative finance using NVIDIA NeMo Agent Toolkit. This workflow demonstrates how to leverage LLMs to automatically generate, code, and evaluate alpha signals.

## Overview

Signal discovery is the process of finding quantitative signals — also known as alpha signals — that have predictive power for future stock returns. This workflow automates the traditional labor-intensive process using LLMs.

### Workflow Architecture

![Workflow Architecture](notebooks/images/workflow-architecture.png)

The workflow uses a **closed-loop optimization** approach:
1. Generate signal ideas using an LLM
2. Convert ideas to executable Python code
3. Evaluate the signal's predictive power (Rank IC)
4. If IC meets threshold → Accept and save
5. If IC is poor → Generate optimization advice and retry

## Getting Started

### Prerequisites

- **Platform:** Linux, macOS, or Windows
- **Python:** version 3.11, 3.12, or 3.13
- **Package manager:** pip or uv

### API Keys

You will need an NVIDIA API key. Get yours from [build.nvidia.com](https://build.nvidia.com/settings/api-keys).

```bash
export NVIDIA_API_KEY="your-api-key-here"
```

### Installing Dependencies

```bash
uv sync --locked
```

(All commands below use `uv run ...` so you don't need to activate the venv. If you prefer to activate it once and drop the `uv run` prefix, run `source .venv/bin/activate` first.)

### Download Data

The workflow requires S&P 500 price-volume data (Open, Close, High, Low, Volume). Use the included script to download fresh data via [yfinance](https://github.com/ranaroussi/yfinance):

```bash
uv run python -m signal_discovery_workflow.download_data
```

You can customize the date range:

```bash
uv run python -m signal_discovery_workflow.download_data --start 2015-01-01 --end 2025-12-31
```

#### Local fork: universe presets (Tier 1 expansion)

> This subsection documents a local modification of the upstream example.
> With no arguments, behavior is identical to upstream (S&P 500 into `data/sp500/`).

The download script ships presets beyond S&P 500, all tradable panels via yfinance (all symbols with ≥10 instruments, satisfying the cross-sectional Rank IC minimum):

| `--universe` | Panel | Notes |
|---|---|---|
| `sp500` (default) | S&P 500 constituents | Upstream behavior |
| `nasdaq100` | Curated Nasdaq-100 snapshot | Listed snapshot; refresh periodically |
| `dow30` | Dow Jones Industrial Average 30 | 29–30 names; noisy IC |
| `country_etfs` | 20 single-country MSCI ETFs (EWA…EIDO) | "Indices as instruments" — a single ^index cannot yield a cross-sectional Rank IC |
| `sector_etfs` | 10 SPDR sector ETFs | XLRE has shorter history (2015+) |
| `bond_etfs` | 12 rates/credit/TIPS/muni ETFs | Duration & credit panel; HY/EM/muni from 2006–07 |
| `factor_etfs` | 10 factor ETFs (MTUM, QUAL, …) | Style-premium panel |

Custom universes are supported with a comma-separated ticker list (writes to `data/custom/` by default):

```bash
uv run python -m signal_discovery_workflow.download_data --tickers "AAPL,MSFT,NVDA,AMZN"
```

Point the workflow at a downloaded universe via the `SIGNAL_DATA_DIR` environment variable (defaults to `data/sp500/` when unset):

```powershell
# PowerShell
$env:SIGNAL_DATA_DIR = "src/signal_discovery_workflow/data/nasdaq100"
uv run nat run --config_file configs/config-optimization.yml --input "momentum signals"

# bash
SIGNAL_DATA_DIR=src/signal_discovery_workflow/data/nasdaq100 uv run nat run --config_file configs/config-optimization.yml --input "momentum signals"
```

> **Disclaimer:** Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use.

## Deployment Options

This workflow can be deployed in three ways:

### Deploy on Brev

Spin up a one-click GPU instance on [brev.nvidia.com](https://brev.nvidia.com/) and open [`brev/launchable-setup.ipynb`](brev/launchable-setup.ipynb). It clones the repo, installs `uv`, syncs dependencies, and registers a Jupyter kernel. When it finishes, follow the inline link to jump into [`brev/signal-discovery-workflow.ipynb`](brev/signal-discovery-workflow.ipynb), which uses [`configs/config-optimization-no-telemetry.yml`](configs/config-optimization-no-telemetry.yml) so the cloud notebook does not require a separate Phoenix server.

### Option 1: Interactive Notebook Deployment

Best for exploration, experimentation, and learning. The notebook provides step-by-step execution with inline documentation.

```bash
uv run jupyter notebook notebooks/signal-discovery-workflow.ipynb
```

The notebook includes:
- API key setup
- Configuration exploration
- Step-by-step workflow execution
- Interactive result visualization
- Ability to modify parameters on-the-fly

### Option 2: CLI Deployment

Best for production, automation, and scripting. Run the workflow directly from the command line.

#### Basic Usage

```bash
# Run the signal discovery workflow
uv run nat run --config_file configs/config-optimization.yml --input "momentum signals"
```

#### With Phoenix Tracing (Recommended)

For full observability with LLM tracing, run Phoenix in a separate terminal first:

**Terminal 1 - Start Phoenix Server:**
```bash
uv run phoenix serve
```

Phoenix will start at http://localhost:6006

**Terminal 2 - Run the Workflow:**
```bash
export NVIDIA_API_KEY="your-api-key-here"
uv run nat run --config_file configs/config-optimization.yml --input "momentum signals"
```

View traces at http://localhost:6006 to see:
- LLM calls and responses
- Token usage
- Latency metrics
- Full execution trace

#### Running Different Signal Types

```bash
# Generate volatility signals
uv run nat run --config_file configs/config-optimization.yml --input "volatility signals"

# Generate mean reversion signals
uv run nat run --config_file configs/config-optimization.yml --input "mean reversion signals"

# Generate volume-based signals
uv run nat run --config_file configs/config-optimization.yml --input "volume price divergence signals"
```

## Components

| Component | Description |
|-----------|-------------|
| **Signal Agent** | Uses an LLM to generate signal expressions based on price-volume data and operators |
| **Code Agent** | Wraps each signal formula in a Python function via an LLM, and inlines the required operator implementations from `calculator.json` to produce a self-contained executable module |
| **Eval Agent** | Performs backtesting via Rank IC and generates optimization suggestions |
| **Data Download Script** | Fetches S&P 500 price-volume data from Yahoo Finance via `yfinance` |

## Configuration

The workflow configuration is defined in `configs/config-optimization.yml`:

> **Note:** The `base_url` for the LLMs depends on your API key. Set it to either:
> - `https://integrate.api.nvidia.com/v1/` — for keys from [build.nvidia.com](https://build.nvidia.com)
> - `https://inference-api.nvidia.com/v1/` — for NVIDIA internal or enterprise API keys

| Parameter | Description |
|-----------|-------------|
| `signal_generator_llm` | Reference to the LLM block used for signal ideation (typically higher temperature for creativity) |
| `code_generator_llm` | Reference to the LLM block used for translating formulas into Python (low temperature for determinism) |
| `optimization_advisor_llm` | Reference to the LLM block used to produce iteration feedback (balanced temperature) |
| `ic_threshold` | Minimum absolute IC value to accept a signal (e.g., 0.02 = 2%) |
| `p_value_threshold` | Maximum p-value for statistical significance (e.g., 0.05 = 5%) |
| `max_iterations` | Maximum number of optimization iterations before returning the best result |
| `num_signals` | Number of signals to generate per iteration |
| `forward_periods` | Number of days for forward return calculation (e.g., 5 = weekly) |
| `save_results` | Whether to save accepted/best-effort signals to `output/` |

You can use the same model for all three agents, or assign separate models to the Signal, Code, and Advisor agents by editing the `llms` blocks in the YAML.

## Evaluation Metrics

The workflow uses two key metrics to decide whether to accept or reject a generated signal:

| Metric | Description | Acceptance Criteria |
|--------|-------------|---------------------|
| **Mean IC** | Average Spearman rank correlation between signal values and forward returns, computed across all time periods | \|IC\| ≥ `ic_threshold` (default: 0.02) |
| **P-value** | Statistical significance of the mean IC being different from zero | ≤ `p_value_threshold` (default: 0.05) |

A signal is accepted when both criteria are met. Otherwise, the Eval Agent generates optimization suggestions and the workflow retries.

## Workflow Result Format

Each run returns a structured JSON result containing the outcome, metrics, the signals that were tried, and (if the loop did not accept on the first iteration) the optimization advice produced for the next attempt. Example:

```json
{
  "status": "best_effort",
  "headline": "Best-effort result (IC threshold not met)",
  "request": "momentum signals",
  "iteration": 2,
  "total_iterations": 3,
  "selected_signal": "signal_volume_decayed_momentum",
  "thresholds": {
    "ic_threshold": 0.02,
    "p_value_threshold": 0.05
  },
  "metrics": {
    "mean_ic": 0.0103,
    "ic_std": 0.21,
    "ic_ir": 0.049,
    "t_stat": 2.91,
    "p_value": 0.0036,
    "num_periods": 3494,
    "positive_ic_ratio": 0.529
  },
  "signals": [
    {
      "name": "Volume-Decayed Momentum",
      "formula": "Mul(TS_Return(Close, 20), Decay_Linear(Volume, 20))",
      "category": "momentum",
      "data_fields_used": ["Close", "Volume"],
      "lookback_periods": [20]
    }
  ],
  "saved_path": "src/signal_discovery_workflow/output/signal_xxx.json",
  "last_feedback": "- Try TS_Std instead of TS_Var for cleaner volatility signal\n- Use 60-day lookback instead of 20\n- Replace Volume with Close*Volume for dollar-volume weighting"
}
```

| Field | Description |
|-------|-------------|
| `status` | `"accepted"`, `"best_effort"`, or `"failed"` |
| `headline` | Human-readable summary |
| `iteration` / `total_iterations` | Which iteration produced this result, out of the max allowed |
| `selected_signal` | Python function name of the signal whose IC was reported (the evaluator picks the best when multiple signals are returned) |
| `metrics` | All non-null IC statistics |
| `signals` | Compact summary of every signal that was generated |
| `saved_path` | Where the full signal JSON + code was persisted |
| `last_feedback` | Optimization advice from the last failed iteration; pass it back to resume the loop |

### Resuming an Optimization Loop

The workflow input accepts either a plain string or a JSON object that bundles `seed_feedback` from a prior run. Pass the `last_feedback` field from a prior result to start a new loop with the previous advice already applied — useful when you want more iterations than `max_iterations` allows, or want to switch models mid-run.

The shell snippet below uses `jq` to read `last_feedback` from the prior result and pack it into the JSON input shape (install with `brew install jq` on macOS or `apt-get install jq` on Debian/Ubuntu):

```bash
# First run - best effort, did not converge
uv run nat run --config_file configs/config-optimization.yml --input "momentum signals" > result1.json

# Extract last_feedback and resume
SEED=$(jq -r '.last_feedback' result1.json)
uv run nat run --config_file configs/config-optimization.yml \
  --input "$(jq -nc --arg req 'momentum signals' --arg seed "$SEED" \
              '{request: $req, seed_feedback: $seed}')"
```

Or programmatically, passing the same JSON shape as the workflow's input string:

```python
import json

result1 = json.loads(await runner.ainvoke("momentum signals"))

resume_input = json.dumps({
    "request": "momentum signals",
    "seed_feedback": result1["last_feedback"],
})
result2 = json.loads(await runner.ainvoke(resume_input))
```

The `last_feedback` field can be persisted to disk and re-loaded later — there's no in-memory state required to resume.

## Development

Tests cover the agent helpers, prompt builders, JSON parsing, module assembly, and end-to-end execution:

```bash
uv sync --locked --extra test
uv run --locked pytest tests/
```

The repo also ships a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs `ruff` lint and the test suite on every pull request to `main`.

## Project Structure

```
quantitative-signal-discovery-agent/
├── .github/workflows/ci.yml             # PR-level CI: ruff lint + pytest
├── brev/
│   └── launchable-setup.ipynb           # One-shot environment setup for Brev launchables
├── configs/
│   ├── config-optimization.yml          # Workflow + LLM config with Phoenix tracing
│   └── config-optimization-no-telemetry.yml # Brev/cloud config, no Phoenix tracing
├── notebooks/
│   ├── signal-discovery-workflow.ipynb  # Interactive walkthrough
│   └── images/workflow-architecture.png
├── pyproject.toml                       # Dependencies, ruff/pytest config, NAT entry point
├── uv.lock                              # Pinned dependency resolution
├── README.md
├── src/signal_discovery_workflow/
│   ├── __init__.py
│   ├── register.py                                 # NAT function registration
│   ├── signal_generator.py                         # Signal agent: generates JSON signal descriptions
│   ├── signal_code_generator.py                    # Code agent: turns JSON into executable Python
│   ├── signal_evaluator.py                         # Eval agent: runs signal code, computes Rank IC
│   ├── signal_discovery_optimization_workflow.py   # Orchestrator (closed-loop generate/code/eval/feedback)
│   ├── llm_utils.py                                # Shared LLM-output helpers (parse, sanitize, normalize)
│   ├── download_data.py                            # Fetches S&P 500 data via yfinance
│   ├── data/sp500/                                 # OHLCV CSVs (gitignored)
│   ├── output/                                     # Saved signal results (gitignored)
│   └── template/
│       ├── calculator.json                         # Operator catalogue (name, signature, code)
│       └── signal_output_template.json             # JSON schema the signal agent fills in
└── tests/                                          # pytest suite
```

## Additional Resources

- [NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo-agent-toolkit/)
- [Arize Phoenix Documentation](https://arize.com/docs/phoenix)
- [NeMo Fine-tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html) — to specialize Nemotron on your signal history

## Local Fork: Experiment Results (2026-11-12)

All runs: hosted Nemotron 3.5 Lightning via build.nvidia.com, CPU-only local evaluation, Windows. Full detail in `docs/EXPERIMENTS.md`.

| Universe | Request | Status | |IC| | Note |
|---|---|---|---|---|---|---|
| bond_etfs | momentum | **accepted** | 0.0322 | First-iteration accept; reports to `signal_20261112_053559_iter1_report.md` |
| sp500 | mean reversion | best_effort | 0.0167 | t=−6.3; nearest miss |
| sp500 | momentum | best_effort | 0.0119 | baseline |
| sp500 | volatility | best_effort | 0.0048 | weak |

Key findings: identical `Div(TS_Return(Close, 20), TS_Std(Close, 20))` formula passed on bond ETFs and failed on mega-cap equities; every promising IC was negative-signed (inverse edge); Newey–West-adjusted reports show the bond and reversion signals remain significant (p = 0.009 / 2e-04), but cost-aware long-short spreads are negative as coded — `|IC|` acceptance is necessary, not sufficient. Operator-hallucination aliases were added in `llm_utils.OPERATOR_SYNONYMS` and `signal_report.py` adds HAC-adjusted IC, decile long-short economics, turnover cost drag and sub-period stability.

New toolkit on top of upstream:

```bash
# Deep-validate any saved signal (HAC stats, cost-aware backtest, stability)
uv run python -m signal_discovery_workflow.signal_report src/signal_discovery_workflow/output/signal_xxx.json --data-dir src/signal_discovery_workflow/data/bond_etfs
```

## Local Fork Modifications

This checkout carries additive-only changes on branch `local/tier1-universes`. Defaults are byte-identical to upstream; every new capability is opt-in.

- `download_data.py`: added `--universe` presets (Nasdaq 100, Dow 30, country/sector/bond/factor ETFs) and `--tickers` for custom panels; new `download_universe_data()`; `download_sp500_data()` retained as a backwards-compatible wrapper.
- `signal_evaluator.py`: `load_stock_data()` accepts an optional `data_dir` and honors the `SIGNAL_DATA_DIR` env var (default: upstream `data/sp500/`).
- `llm_utils.py`: `OPERATOR_SYNONYMS` extended with near-miss hallucination aliases observed in real runs (e.g. `TS_Volatility→TS_Vol`, `TS_Kurtosis→TS_Kurt`, `EWM_Mean→EMA`). Zero exact matches changed.
- `signal_report.py` (new): accuracy/backtest report for saved signals — daily Rank IC series, Newey-West HAC t/p, decile long-short spread with turnover and cost drag, sub-period stability, max drawdown. Pure offline computation; no workflow changes.
- `.gitignore`: downloaded CSVs and `output/` artifacts ignored for all universes.
- No changes to agents, prompts, operator catalogue, Rank IC math, thresholds, YAML configs, or NAT registration.

## License

See [LICENSE.txt](LICENSE.txt) and [LICENSE-3rd-party.txt](LICENSE-3rd-party.txt) for details.
