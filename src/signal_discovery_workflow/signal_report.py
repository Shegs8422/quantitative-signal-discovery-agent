# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# The accuracy/backtest tooling below is a local-fork addition and is not
# part of the upstream example. Review the NVIDIA Software License Agreement
# and Product Specific Terms for AI Products before redistribution.

"""
Signal accuracy report (local fork, branch feat/tier1-universes-assurance).

Deep validation for a signal previously produced by the workflow. Adds:

- Daily per-period Spearman Rank IC series
- Newey-West (HAC) t-stat / p-value, lag = forward periods
- Moving-block bootstrap confidence interval for mean IC (small panels)
- Horizon ladder: IC @ 1/5/10/21/63d + signal lag-1..5 autocorrelation
- Regime splits: mean IC by calendar year and by trailing-vol tercile
- Residualized (market-beta-neutral) IC vs equal-weight panel, trailing-126d
- Deflated inference: trial registry (output/trials.jsonl) + Bonferroni alpha
- Cross-sectional top/bottom decile (or quintile for small panels) long-short
  backtest with turnover and one-way transaction-cost drag
- Sub-period stability, positive-spread rate, max drawdown
- Optional FX attribution for international panels (--local-currency), which
  re-evaluates the signal on local-currency returns using FX pair CSVs

This module is additive: it reads the same OHLCV CSVs and the same saved
signal code the workflow already persists. It does not alter the workflow,
its scoring, or its acceptance criteria.

Usage:
    python -m signal_discovery_workflow.signal_report output/signal_xxx.json
    python -m signal_discovery_workflow.signal_report output/signal_xxx.json --data-dir path/to/data/sp500
    python -m signal_discovery_workflow.signal_report output/signal_xxx.json --local-currency --fx-data-dir data/fx
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

# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


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


def _block_bootstrap_ci(
    x: np.ndarray, n_iter: int = 1000, block: int = 10, seed: int = 42
) -> tuple[float, float] | None:
    """Moving-block bootstrap 95% CI for the mean. Honest for small panels/overlap."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, n - block, size=n // block + 1)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _annualized_stats(spread: pd.Series) -> dict[str, float | None]:
    """Naive annualization (overlap-inflated; flagged everywhere it's used)."""
    spread = spread.dropna()
    n = len(spread)
    if n < 30:
        return {"ann_return": None, "ann_vol": None, "sharpe": None}
    ann_return = float(spread.mean() * 252)
    ann_vol = float(spread.std() * math.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else None
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe}


# ---------------------------------------------------------------------------
# Long-short backtest
# ---------------------------------------------------------------------------


def _decile_long_short(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Daily top-vs-bottom decile (equal-weight) spread forward returns.

    Deciles when n >= 30 instruments that day, else top/bottom ~20%.
    Returns DataFrame[date, spread, turnover, net_spread] sorted by date.
    """
    common_dates = signal_values.index.intersection(forward_returns.index)
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


# ---------------------------------------------------------------------------
# Horizon ladder, autocorrelation, regimes, residualized IC
# ---------------------------------------------------------------------------


def _horizon_ladder(
    signal_values: pd.DataFrame,
    close: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 63),
) -> dict[int, dict]:
    """Mean IC / HAC stats per forward-return horizon."""
    out = {}
    for h in horizons:
        fwd = compute_forward_returns(close, periods=h)
        ic = _daily_rank_ic_series(signal_values, fwd)
        mean, t, p = _newey_west_stats(ic.to_numpy(), lag=h)
        out[h] = {"n": len(ic), "mean_ic": mean, "hac_t": t, "hac_p": p}
    return out


def _signal_autocorr(signal_values: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 5)) -> dict[int, float]:
    """Median per-instrument lag-k autocorrelation of the signal."""
    out: dict[int, float] = {}
    for k in lags:
        corrs = []
        for col in signal_values.columns:
            s = signal_values[col].dropna()
            if len(s) < k + 50:
                continue
            a, b = s.iloc[:-k], s.iloc[k:]
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
        out[k] = float(np.nanmedian(corrs)) if corrs else float("nan")
    return out


def _panel_returns(close: pd.DataFrame) -> pd.Series:
    """Equal-weight panel daily return (our market proxy when no index is provided)."""
    return close.pct_change().mean(axis=1)


def _regime_splits(ic_series: pd.Series, close: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean IC by calendar year, and by trailing-21d realized panel-vol tercile."""
    yearly: dict[str, float] = {}
    for yr in sorted(set(ic_series.index.year)):
        part = ic_series[ic_series.index.year == yr]
        yearly[str(yr)] = float(part.mean()) if len(part) >= 20 else float("nan")

    panel_ret = _panel_returns(close)
    vol = panel_ret.rolling(21).std()
    vol_t = vol.quantile([1 / 3, 2 / 3])
    regime: dict[str, float] = {}
    tercile = pd.cut(vol, [-np.inf, vol_t.iloc[0], vol_t.iloc[1], np.inf], labels=["low_vol", "mid_vol", "high_vol"])
    for label in ("low_vol", "mid_vol", "high_vol"):
        dates = tercile[tercile == label].index
        part = ic_series.loc[ic_series.index.intersection(dates)]
        regime[label] = float(part.mean()) if len(part) >= 20 else float("nan")
    return {"by_year": yearly, "by_vol_tercile": regime}


def _residualized_ic(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    close: pd.DataFrame,
    lookback: int = 126,
    periods: int = 5,
) -> dict:
    """
    Market-beta-neutral Rank IC: residualize each instrument's forward return
    against the equal-weight panel using trailing-`lookback`d betas.
    """
    panel_ret = _panel_returns(close)
    panel_close = close.mean(axis=1)
    panel_fwd = panel_close.shift(-periods) / panel_close - 1

    ret_daily = close.pct_change()
    residuals = []
    for col in close.columns:
        beta = ret_daily[col].rolling(lookback).cov(panel_ret) / panel_ret.rolling(lookback).var()
        resid_fwd = forward_returns[col] - beta.reindex(forward_returns.index) * panel_fwd
        residuals.append(resid_fwd.rename(col))
    resid_df = pd.concat(residuals, axis=1)
    ic = _daily_rank_ic_series(signal_values, resid_df)
    mean, t, p = _newey_west_stats(ic.to_numpy(), lag=periods)
    return {"n": len(ic), "mean_ic": mean, "hac_t": t, "hac_p": p}


# ---------------------------------------------------------------------------
# Trial registry (deflated inference) and FX attribution
# ---------------------------------------------------------------------------


def _record_trial(signal_json_path: str, mean_ic: float | None, hac_p: float | None) -> int:
    """Append this evaluation to output/trials.jsonl and return total N."""
    trials_path = Path(signal_json_path).parent / "trials.jsonl"
    n = 0
    try:
        if trials_path.exists():
            n = sum(1 for line in trials_path.read_text(encoding="utf-8").splitlines() if line.strip())
        n += 1
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "signal_json": Path(signal_json_path).name,
            "mean_ic": mean_ic,
            "hac_p_value": hac_p,
        }
        with trials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:
        logger.warning(f"trial registry unavailable: {e}")
        n = n or 1
    return n


# Country ETF -> (yf symbol for FX pair, invert to USD-per-CCY)
FX_COUNTRY_MAP: dict[str, tuple[str, bool]] = {
    "EWA": ("AUDUSD=X", False),
    "EWC": ("USDCAD=X", True),
    "EWD": ("USDSEK=X", True),
    "EWG": ("EURUSD=X", False),
    "EWH": ("USDHKD=X", True),
    "EWI": ("EURUSD=X", False),
    "EWJ": ("USDJPY=X", True),
    "EWL": ("USDCHF=X", True),
    "EWN": ("EURUSD=X", False),
    "EWP": ("EURUSD=X", False),
    "EWQ": ("EURUSD=X", False),
    "EWY": ("USDKRW=X", True),
    "EWZ": ("USDBRL=X", True),
    "EWT": ("USDTWD=X", True),
    "THD": ("USDTHB=X", True),
    "INDA": ("USDINR=X", True),
    "FXI": ("USDCNY=X", True),
    "MCHI": ("USDCNY=X", True),
    "EPOL": ("USDPLN=X", True),
    "EIDO": ("USDIDR=X", True),
}


def _fx_rates(close: pd.DataFrame, fx_dir: str | Path) -> tuple[dict[str, pd.Series], list[str]]:
    """Load USD-per-CCY rate series for recognized country ETFs."""
    rates: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for etf, (symbol, invert) in FX_COUNTRY_MAP.items():
        if etf not in close.columns:
            continue
        path = Path(fx_dir) / f"{symbol}.csv"
        if not path.exists():
            skipped.append(f"{etf} ({symbol}): missing {path.name}")
            continue
        try:
            fx = pd.read_csv(path, index_col=0, parse_dates=True)
            series = fx.iloc[:, 0].dropna()
            if invert:
                series = 1.0 / series
            series = series.reindex(close.index).ffill()
            if series.isna().all():
                skipped.append(f"{etf} ({symbol}): FX series empty over sample")
                continue
            rates[etf] = series
        except Exception as e:
            skipped.append(f"{etf} ({symbol}): {e}")
    return rates, skipped


def _local_stock_data(
    stock_data: dict[str, pd.DataFrame], fx_dir: str | Path
) -> tuple[dict[str, pd.DataFrame], int, list[str]]:
    """
    Re-express OHLC prices in local currency (local = USD / F, F = USD per CCY).
    Volume (shares) is unchanged. Returns (local_data, converted_count, skipped).
    """
    rates, skipped = _fx_rates(stock_data["Close"], fx_dir)
    local_data: dict[str, pd.DataFrame] = {}
    for field, df in stock_data.items():
        if field in ("Open", "Close", "High", "Low"):
            df = df.copy()
            for etf, f_rate in rates.items():
                if etf in df.columns:
                    df[etf] = df[etf] / f_rate
            local_data[field] = df
        else:
            local_data[field] = df
    return local_data, len(rates), skipped


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.4f}"


def _fmt2(v: float | None, spec: str = ".2f") -> str:
    return "n/a" if v is None else format(v, spec)


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


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
            periods: int = 5, cost_bps: float = 10.0,
            local_currency: bool = False, fx_data_dir: str | Path | None = None,
            skip_expensive: bool = False) -> str:
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

    close = stock_data["Close"]
    forward_returns = compute_forward_returns(close, periods=periods)
    ic_series = _daily_rank_ic_series(signal_values, forward_returns)
    ls = _decile_long_short(signal_values, forward_returns, cost_bps=cost_bps)
    ls_inv = _decile_long_short(-signal_values, forward_returns, cost_bps=cost_bps)

    ic_mean, ic_hac_t, ic_hac_p = _newey_west_stats(ic_series.to_numpy(), lag=periods)
    if ic_mean is None:
        ic_mean = float("nan")
    ic_ci = _block_bootstrap_ci(ic_series.to_numpy())

    bottom = f"{_fmt2(ic_ci[0], '+.4f')} to {_fmt2(ic_ci[1], '+.4f')}" if ic_ci else "n/a (sample too small)"

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
    inv_net_mean = float(ls_inv["net_spread"].mean()) if len(ls_inv) else None
    spread_mean_s = _fmt(spread_mean) if spread_mean is not None else "n/a"

    n_inst = close.shape[1]
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
    direction = 1 if (ic_mean or 0) >= 0 else -1

    # Deflated inference (trial registry)
    n_trials = _record_trial(signal_json_path, ic_mean, ic_hac_p)
    bonf_alpha = 0.05 / n_trials
    bonf_verdict = (
        f"passes Bonferroni (p {ic_hac_p:.4g} ≤ {bonf_alpha:.4g})"
        if ic_hac_p is not None and ic_hac_p <= bonf_alpha
        else ("does NOT pass Bonferroni" if ic_hac_p is not None else "untestable (HAC n/a)")
    )

    # Expensive add-ons (skippable for quick checks)
    horizon_block = ""
    regime_block = ""
    resid_block = ""
    autocorr_block = ""
    if not skip_expensive:
        h = _horizon_ladder(signal_values, close, horizons=(1, 5, 10, 21, 63))
        horizon_block = "\n## Horizon ladder (mean IC by forward horizon)\n" + "\n".join(
            f"- {p}d: IC {_fmt(h[p]['mean_ic'])} (HAC t {_fmt2(h[p]['hac_t'])}, p {_fmt2(h[p]['hac_p'], '.3g')}, n={h[p]['n']})"
            for p in (1, 5, 10, 21, 63)
        )

        autocorr = _signal_autocorr(signal_values)
        autocorr_block = (
            "\n## Signal autocorrelation (median per-instrument)\n"
            + "\n".join(f"- lag {k}d: {v:+.3f}" for k, v in autocorr.items())
        )

        regimes = _regime_splits(ic_series, close)
        regime_block = (
            "\n## Regime splits\n### By calendar year\n"
            + "\n".join(f"- {y}: IC {_fmt(v)}" for y, v in regimes["by_year"].items())
            + "\n### By trailing-21d panel realized-vol tercile\n"
            + "\n".join(f"- {k}: IC {_fmt(v)}" for k, v in regimes["by_vol_tercile"].items())
        )

        resid = _residualized_ic(signal_values, forward_returns, close, lookback=126, periods=periods)
        resid_block = (
            "\n## Market-neutral IC (residualized vs equal-weight panel, trailing-126d beta)\n"
            f"- Mean IC: {_fmt(resid['mean_ic'])} | HAC t: {_fmt2(resid['hac_t'])} | p = {_fmt2(resid['hac_p'], '.3g')} | n = {resid['n']}\n"
            "(Subtracts per-instrument rolling beta × EW panel forward return before correlating.)"
        )

    # FX attribution (--local-currency): re-execute the SAME saved signal module
    # on local-currency panels so signal, prices and returns are all in local terms.
    fx_block = ""
    if local_currency:
        if not fx_data_dir:
            fx_block = "\n## FX attribution\nRequested --local-currency but no --fx-data-dir provided.\n"
        else:
            local_data, converted, skipped = _local_stock_data(stock_data, fx_data_dir)
            if converted < 3:
                fx_block = (
                    f"\n## FX attribution\nOnly {converted} instrument(s) could be converted — need >= 3. "
                    f"Skipped: {'; '.join(skipped) or 'all'}\n"
                )
            else:
                local_exec = execute_signal_code(saved["signal_code"], local_data, selection_periods=periods)
                if local_exec is None:
                    fx_block = "\n## FX attribution\nSignal code failed to execute on local-currency panel.\n"
                else:
                    sig_local, _name = local_exec
                    local_fwd = compute_forward_returns(local_data["Close"], periods=periods)
                    local_ic = _daily_rank_ic_series(sig_local, local_fwd)
                    l_mean, l_t, l_p = _newey_west_stats(local_ic.to_numpy(), lag=periods)
                    ls_local = _decile_long_short(sig_local, local_fwd, cost_bps=cost_bps)
                    l_net = float(ls_local["net_spread"].mean()) if len(ls_local) else None
                    fx_block = (
                        "\n## FX attribution (USD vs local currency)\n"
                        f"- Converted instruments: {converted} (skipped: {'; '.join(skipped) or 'none'})\n"
                        f"- USD-denominated:   mean IC {_fmt(ic_mean)} | net L/S spread {_fmt(net_mean)}\n"
                        f"- Local-currency:    mean IC {_fmt(l_mean)} (HAC t {_fmt2(l_t)}, p {_fmt2(l_p, '.3g')}) | "
                        f"net L/S spread {_fmt(l_net)}\n"
                        "- Note: USD returns embed FX. If local IC ≈ USD IC, the edge is asset selection; "
                        "if local IC collapses to ~0, the 'alpha' was largely the currency."
                    )

    md = f"""# Signal Accuracy Report — `{p_name}`

- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
- Signal (selected): `{selected_name}` (best of functions defined in the saved module)
- Category: {_category_of(saved)}
- Forward horizon: {periods} periods (daily bars)
- Sample: {len(ic_series)} cross-sections, {n_inst} instruments
- Recommended direction: {'long the coded formula (+1)' if direction == 1 else 'flip the coded formula (-1, inverse-predictive as written)'}

## Rank IC (daily cross-sectional Spearman)
- Mean IC: **{ic_mean:+.4f}** | Bootstrap 95% CI: {bottom}
- IC std: {ic_series.std():.4f}
- Naive t (independence, what the workflow reports): {naive_t:+.2f}
- **Newey-West (HAC, lag={periods}) t: {_fmt2(ic_hac_t)}, p = {_fmt2(ic_hac_p, '.4g')}**
- Positive-IC ratio: {_pct(float((ic_series > 0).mean()))}

## Sub-period stability
- First half ({ic_first.index.min():%Y-%m-%d} → {ic_first.index.max():%Y-%m-%d}): mean IC {ic_first.mean():+.4f} (n={len(ic_first)})
- Second half ({ic_second.index.min():%Y-%m-%d} → {ic_second.index.max():%Y-%m-%d}): mean IC {ic_second.mean():+.4f} (n={len(ic_second)})
{horizon_block}
{autocorr_block}
{regime_block}
{resid_block}
{fx_block}
## Decile long-short backtest ({k_label}, equal weight, {bucket_n} per leg)
- Gross mean spread: **{spread_mean_s}** per {periods}-day period
- Newey-West t: {_fmt2(spread_hac_t)}, p = {_fmt2(spread_hac_p, '.4g')}
- Positive spread periods: {_pct(positive_rate)}
- Avg turnover per side: {_pct(avg_turnover)} | Cost drag: {cost_bps} bps one-way → mean cost ≈ {_fmt2(cost_mean, '.5f')} per period
- **Net mean spread (as coded): {_fmt(net_mean)}**
- Inverse direction net spread: {_fmt(inv_net_mean)}
- Max drawdown of cumulative spread: {_fmt2(max_drawdown, '.4f')}
- Naive annualized (overlap-inflated) return/vol/Sharpe gross: {_fmt(naive['ann_return'])} / {_fmt(naive['ann_vol'])} / {_fmt(naive['sharpe'])}
- Naive annualized net: {_fmt(net_naive['ann_return'])} / {_fmt(net_naive['ann_vol'])} / {_fmt(net_naive['sharpe'])}

## Multiple-testing context (trial registry — every report run counts)
- Trials to date (incl. this one): **{n_trials}**
- Bonferroni alpha: 0.05 / {n_trials} = {bonf_alpha:.4g}
- Verdict: {bonf_verdict}

## Honest caveats
- The workflow's IC t-stat assumes independent samples; with overlapping {periods}-day forward returns, effective information is ~{periods}× lower. Trust the HAC numbers.
- Overlapping daily spreads, so annualized figures above are inflated and shown only for reference.
- IC sign matters: the workflow accepts on |IC|. A negative mean IC implies the coded signal is *inverse* predictively — direction must be fixed at execution time.
- No slippage model, no market-impact model, no borrow cost for short legs.
- Instrument universe is whatever was in the data dir at report time (check for survivorship bias; constituent lists here are current, not point-in-time).
- Residualized IC uses trailing rolling betas — a mechanical hedge ratio, not a full risk model.
"""
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy report for a saved workflow signal.")
    parser.add_argument("signal_json", help="Path to the saved signal JSON (src/.../output/signal_*.json)")
    parser.add_argument("--data-dir", default=None, help="Override OHLCV CSV directory (default: SIGNAL_DATA_DIR or data/sp500)")
    parser.add_argument("--forward-periods", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0, help="One-way transaction cost in bps (default 10)")
    parser.add_argument("--local-currency", action="store_true", help="Add FX-attribution block for country-ETF universes")
    parser.add_argument("--fx-data-dir", default=None, help="Directory with FX pair CSVs (see download_data.py --download-fx)")
    parser.add_argument("--skip-expensive", action="store_true", help="Skip ladder/regime/residualized blocks (faster)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    md = analyze(
        args.signal_json,
        args.data_dir,
        periods=args.forward_periods,
        cost_bps=args.cost_bps,
        local_currency=args.local_currency,
        fx_data_dir=args.fx_data_dir,
        skip_expensive=args.skip_expensive,
    )

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
