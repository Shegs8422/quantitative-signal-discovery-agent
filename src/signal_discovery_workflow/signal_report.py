# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# SPDX-License-Identifier: Apache-2.0

"""
Signal accuracy report (local fork, branch local/tier1-universes).

Deep validation for a signal previously produced by the workflow. The built-in
evaluator reports Rank IC with a naive, independence-assuming t-stat and no
portfolio-level statistics. Overlapping forward returns (e.g. 5-day horizon
sampled daily) violate that assumption, so this report adds:

- Daily per-period Spearman Rank IC series
- Newey-West (HAC) t-stat / p-value, lag = forward periods
- Cross-sectional decile long-short spread backtest (equal weight)
- Per-period turnover and a one-way transaction-cost drag (default 10 bps)
- Naive annualized Sharpe of the daily spread series (flagged as
  overlap-inflated; the HAC statistics are the honest read)
- Sub-period stability (split the sample in half by date)
- % positive spread periods, worst period, cumulative spread curve stats

This module is additive: it reads the same OHLCV CSVs and the same saved
signal code the workflow already persists. It does not alter the workflow,
its scoring, or its acceptance criteria.

Usage:
    python -m signal_discovery_workflow.signal_report output/signal_xxx.json
    python -m signal_discovery_workflow.signal_report output/signal_xxx.json --data-dir path/to/data
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .signal_evaluator import (
    compute_forward_returns,
    execute_signal_code,
    load_stock_data,
)

logger = logging.getLogger(__name__)


def _daily_rank_ic_series(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Per-date cross-sectional Spearman Rank IC (mirrors the evaluator's loop)."""
    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)
    signal_aligned = signal_values.loc[common_dates, common_stocks]
    returns_aligned = forward_returns.loc[common_dates, common_stocks]

    ics = {}
    for date in common_dates:
        s = signal_aligned.loc[date].dropna()
        r = returns_aligned.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 10:
            continue
        sv, rv = s[common].to_numpy(), r[common].to_numpy()
        if np.std(sv) < 1e-10 or np.std(rv) < 1e-10:
            continue
        rho, _ = stats.spearmanr(sv, rv)
        if not np.isnan(rho):
            ics[date] = float(rho)
    return pd.Series(ics, dtype=float).sort_index()


def _newey_west_stats(x: np.ndarray, lag: int) -> tuple[float, float | None, float | None]:
    """(mean, hac_t_stat, hac_p_value) with Newey-West HAC variance."""
    x = x[~np.isnan(x)]
    n = len(x)
    mean = float(np.mean(x))
    if n <= lag + 2:
        return mean, None, None
    centered = x - mean
    gamma0 = float(np.mean(centered**2))
    var = gamma0
    for i in range(1, lag + 1):
        gamma_i = float(np.mean(centered[i:] * centered[:-i]))
        var += 2.0 * (1.0 - i / (lag + 1)) * gamma_i
    if var <= 0:
        return mean, None, None
    se = math.sqrt(var / n)
    t_stat = mean / se
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    return mean, float(t_stat), p_value


def _decile_long_short(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Daily top-vs-bottom decile (equal-weight) spread forward returns.

    Returns a DataFrame indexed by date with columns:
    ``spread`` (gross), ``turnover`` (avg fraction of sleeve rotated),
    ``net_spread`` (spread minus one-way cost x turnover).
    """
    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)
    rows = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for date in common_dates:
        s = signal_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 6:
            continue
        sv = s[common]
        # Deciles for broad panels; top/bottom ~20% for small ones.
        frac = 0.1 if len(common) >= 30 else 0.2
        k = max(1, int(len(common) * frac))
        ranked = sv.sort_values(ascending=False)
        long_names, short_names = set(ranked.index[:k]), set(ranked.index[-k:])
        spread = float(r[common].loc[list(long_names)].mean() - r[common].loc[list(short_names)].mean())

        if prev_long and prev_short:
            turnover = 1.0 - (len(long_names & prev_long) / k + len(short_names & prev_short) / k) / 2.0
        else:
            turnover = 1.0
        prev_long, prev_short = long_names, short_names

        cost = (cost_bps / 1e4) * turnover
        rows.append((date, spread, turnover, spread - cost))

    df = pd.DataFrame(rows, columns=["date", "spread", "turnover", "net_spread"]).set_index("date")
    for col in ("spread", "turnover", "net_spread"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _annualized_stats(spread: pd.Series) -> dict[str, float | None]:
    """Naive annualization (flagged in the report as overlap-inflated)."""
    spread = spread.dropna()
    n = len(spread)
    if n < 30:
        return {"ann_return": None, "ann_vol": None, "sharpe": None}
    ann_return = float(spread.mean() * 252)
    ann_vol = float(spread.std() * math.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else None
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe}


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.4f}"


def _fmt2(v: float | None, spec: str = ".2f") -> str:
    return "n/a" if v is None else format(v, spec)


def _category_of(saved: dict) -> str:
    try:
        from .llm_utils import extract_json_array

        items = extract_json_array(saved.get("signal_description") or "")
        if not items and saved.get("signal_description"):
            parsed = json.loads(saved["signal_description"])
            items = parsed if isinstance(parsed, list) else [parsed]
        cats = sorted({i.get("category") for i in items if isinstance(i, dict)})
        return ", ".join(c for c in cats if c) or "n/a"
    except Exception:
        return "n/a"


def analyze(signal_json_path: str, data_dir: str | Path | None,
            periods: int = 5, cost_bps: float = 10.0) -> str:
    with open(signal_json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    stock_data = load_stock_data(data_dir)
    if not stock_data:
        raise RuntimeError("No OHLCV data loaded — pass --data-dir or download data first.")

    clean_code = saved["signal_code"]
    exec_result = execute_signal_code(clean_code, stock_data, selection_periods=periods)
    if exec_result is None:
        raise RuntimeError("Signal code failed to execute — see logs above.")
    signal_values, selected_name = exec_result

    forward_returns = compute_forward_returns(stock_data["Close"], periods=periods)
    ic_series = _daily_rank_ic_series(signal_values, forward_returns)
    ls = _decile_long_short(signal_values, forward_returns, cost_bps=cost_bps)

    ic_mean, ic_hac_t, ic_hac_p = _newey_west_stats(ic_series.to_numpy(), lag=periods)

    half = len(ic_series) // 2
    ic_first, ic_second = ic_series.iloc[:half], ic_series.iloc[half:]

    has_ls = len(ls) > 0
    spread_mean, spread_hac_t, spread_hac_p = (
        _newey_west_stats(ls["spread"].to_numpy(), lag=periods) if has_ls else (None, None, None)
    )
    cum_spread = ls["spread"].cumsum() if has_ls else pd.Series(dtype=float)
    max_drawdown = float((cum_spread - cum_spread.cummax()).min()) if has_ls else None
    positive_rate = float((ls["spread"] > 0).mean()) if has_ls else None
    avg_turnover = float(ls["turnover"].mean()) if has_ls else None
    net_mean = float(ls["net_spread"].mean()) if has_ls else None
    cost_mean = (cost_bps / 1e4) * avg_turnover if avg_turnover is not None else None
    spread_mean_s = _fmt(spread_mean) if spread_mean is not None else "n/a"

    n_inst = stock_data["Close"].shape[1]
    k_label = "decile" if n_inst >= 30 else "top/bottom ~20%"
    bucket_n = max(1, int(n_inst * (0.1 if n_inst >= 30 else 0.2)))

    naive = _annualized_stats(ls["spread"])
    net_naive = _annualized_stats(ls["net_spread"])
    naive_t = (
        ic_series.mean() / (ic_series.std() / math.sqrt(len(ic_series)))
        if len(ic_series) and ic_series.std() > 0
        else float("nan")
    )
    p_name = Path(signal_json_path).name
    md = f"""# Signal Accuracy Report — `{p_name}`

- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
- Signal (selected): `{selected_name}` (best of functions defined in the saved module)
- Category: {_category_of(saved)}
- Forward horizon: {periods} periods (daily bars)
- Sample: {len(ic_series)} cross-sections, {stock_data['Close'].shape[1]} instruments

## Rank IC (daily cross-sectional Spearman)
- Mean IC: **{ic_mean:+.4f}**
- IC std: {ic_series.std():.4f}
- Naive t (independence, what the workflow reports): {naive_t:+.2f}
- **Newey-West (HAC, lag={periods}) t: {_fmt2(ic_hac_t)}, p = {_fmt2(ic_hac_p, '.4g')}**
- Positive-IC ratio: {float((ic_series > 0).mean()):.1%}

## Sub-period stability
- First half ({ic_first.index.min():%Y-%m-%d} → {ic_first.index.max():%Y-%m-%d}): mean IC {ic_first.mean():+.4f} (n={len(ic_first)})
- Second half ({ic_second.index.min():%Y-%m-%d} → {ic_second.index.max():%Y-%m-%d}): mean IC {ic_second.mean():+.4f} (n={len(ic_second)})

## Decile long-short backtest ({k_label}, equal weight, {bucket_n} per leg)
- Gross mean spread: **{spread_mean_s}** per {periods}-day period
- Newey-West t: {_fmt2(spread_hac_t)}, p = {_fmt2(spread_hac_p, '.4g')}
- Positive spread periods: {_fmt2(positive_rate, '.1%')}
- Avg turnover per side: {_fmt2(avg_turnover, '.1%')} | Cost drag: {cost_bps} bps one-way → mean cost ≈ {_fmt2(cost_mean, '.5f')} per period
- **Net mean spread: {_fmt(net_mean)}**
- Max drawdown of cumulative spread: {_fmt2(max_drawdown, '.4f')}
- Naive annualized (overlap-inflated) return/vol/Sharpe gross: {_fmt(naive['ann_return'])} / {_fmt(naive['ann_vol'])} / {_fmt(naive['sharpe'])}
- Naive annualized net: {_fmt(net_naive['ann_return'])} / {_fmt(net_naive['ann_vol'])} / {_fmt(net_naive['sharpe'])}

## Honest caveats
- The workflow's IC t-stat assumes independent samples; with overlapping {periods}-day forward returns, effective information is ~{periods}× lower. Trust the HAC numbers.
- Overlapping daily spreads, so annualized figures above are inflated and shown only for reference.
- IC sign matters: the workflow accepts on |IC|. A negative mean IC implies the coded signal is *inverse* predictively — direction must be fixed at execution time.
- No slippage model, no market-impact model, no borrow cost for short legs.
- Instrument universe is whatever was in the data dir at report time (check for survivorship bias).
"""
    return md


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy report for a saved workflow signal.")
    parser.add_argument("signal_json", help="Path to the saved signal JSON (src/.../output/signal_*.json)")
    parser.add_argument("--data-dir", default=None, help="Override OHLCV CSV directory (default: SIGNAL_DATA_DIR or data/sp500)")
    parser.add_argument("--forward-periods", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0, help="One-way transaction cost in bps (default 10)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    md = analyze(args.signal_json, args.data_dir, periods=args.forward_periods, cost_bps=args.cost_bps)

    out = Path(args.signal_json).with_name(Path(args.signal_json).stem + "_report.md")
    out.write_text(md, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(md)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
