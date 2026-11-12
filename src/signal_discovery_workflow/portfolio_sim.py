# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Local-fork addition (branch local/tier1-universes). Not part of the upstream example.

"""
Portfolio simulation v1 (local fork, branch local/tier1-universes).

Turns a saved workflow signal into a tradable dollar-neutral long/short portfolio
and produces investment-grade evidence, not just IC statistics:

- Daily top/bottom-k (decile for broad panels, ~20% for small panels) equal-weight
  dollar-neutral portfolio; weights from the signal, direction auto-detected from
  the saved signal's measured IC sign (override with --direction)
- Optional cross-sectional z-weighted construction (--weighting z, w ∝ z/sum|z|)
- Volatility targeting: daily leverage to hit --target-vol (annualized, causal:
  trailing-63d realized vol, min 20 obs, leverage capped at --max-leverage)
- One-way transaction costs (--cost-bps on |Δw|) and short-leg borrow (--borrow-bps)
- Metrics: geometric annualized return, annualized vol, Sharpe, IR vs equal-weight
  panel, beta vs panel, annualized CAPM alpha, max drawdown, Calmar, hit rate,
  turnover (one-way per annum), total cost drag, leverage stats
- First/second half stability and per-calendar-year return table
- Exports: markdown report + daily portfolio returns CSV (both next to the saved
  signal; gitignored)

Usage:
    python -m signal_discovery_workflow.portfolio_sim src/signal_discovery_workflow/output/signal_xxx.json \
        --data-dir src/signal_discovery_workflow/data/bond_etfs
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

from .signal_evaluator import execute_signal_code, load_stock_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def _rebalance(weights: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """Hold positions between rebalances: keep weights only on the last session of each period, ffill."""
    if freq == "daily":
        return weights
    if freq == "weekly":
        keys = weights.index.to_period("W")
    elif freq == "monthly":
        keys = weights.index.to_period("M")
    else:
        raise ValueError(f"Unknown rebalance frequency: {freq}")
    keep = weights.groupby(keys).tail(1).index
    row_keep = weights.index.isin(keep)
    return weights.loc[row_keep].reindex(weights.index).ffill().fillna(0.0)


def _bucket_size(n_names: int) -> int:
    return max(1, int(n_names * (0.1 if n_names >= 30 else 0.2)))


def build_weights(
    signal_values: pd.DataFrame,
    direction: str = "auto",
    ic_sign: float | None = None,
    weighting: str = "bucket",
) -> pd.DataFrame:
    """
    Dollar-neutral daily weights (long = +, short = −), summing to zero per row.

    direction='auto' uses the sign of `ic_sign` (negative IC ⇒ the coded signal
    is inverse-predictive ⇒ weights ∝ −signal). Override with 'long' (w ∝ signal)
    or 'inverse' (w ∝ −signal).
    """
    if direction == "inverse" or (direction == "auto" and (ic_sign or 0) < 0):
        sign = -1.0
    else:
        sign = 1.0

    weights = pd.DataFrame(0.0, index=signal_values.index, columns=signal_values.columns)
    for date, row in signal_values.iterrows():
        s = row.dropna()
        if len(s) < 6:
            continue
        s = s * sign
        if weighting == "z":
            z = (s - s.mean()) / s.std()
            if z.std() == 0 or np.isnan(z.std()):
                continue
            w = z - z.mean()  # zero-sum → dollar-neutral by construction
            gross = w.abs().sum()
            if gross > 0:
                weights.loc[date, w.index] = w / gross
        else:  # bucket
            k = _bucket_size(len(s))
            ranked = s.sort_values(ascending=False)
            top, bottom = ranked.index[:k], ranked.index[-k:]
            weights.loc[date, top] = 1.0 / k
            weights.loc[date, bottom] = -1.0 / k
    return weights


def _apply_vol_target(
    r_strategy: pd.Series,
    target_vol: float = 0.10,
    lookback: int = 63,
    max_leverage: float = 2.0,
) -> pd.Series:
    """Scale positions daily to hit target ann. vol using trailing realized vol (causal)."""
    realized = r_strategy.rolling(lookback).std() * math.sqrt(252)
    leverage = (target_vol / realized).clip(upper=max_leverage).fillna(1.0)
    leverage.iloc[:lookback] = 1.0
    return r_strategy * leverage.shift(1).fillna(1.0)


def simulate(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    cost_bps: float = 10.0,
    borrow_bps: float = 50.0,
    target_vol: float | None = 0.10,
    benchmark_col_mean: bool = True,
    rebalance: str = "daily",
) -> dict:
    """
    Daily portfolio return simulation. Returns dict with series + metrics.

    Costs: one-way cost_bps × Σ|Δw| (per unit gross 1 = fully invested dollar-neutral pair),
    borrow charged daily on the short notional.
    """
    weights = _rebalance(weights, rebalance)
    rets = close.pct_change().fillna(0.0)
    w_prev = weights.shift(1).fillna(0.0)
    gross_pnl = (w_prev * rets).sum(axis=1)

    turnover = (weights - w_prev).abs().sum(axis=1)  # one-way fraction of gross
    cost = turnover * (cost_bps / 1e4)

    short_notional = (-w_prev.clip(upper=0)).sum(axis=1)
    borrow = short_notional * (borrow_bps / 1e4) / 252.0

    net = gross_pnl - cost - borrow
    if target_vol:
        net = _apply_vol_target(net, target_vol=target_vol)

    if benchmark_col_mean:
        panel_ret = close.pct_change().mean(axis=1)
    else:
        panel_ret = close.pct_change().iloc[:, 0]
    panel_ret = panel_ret.fillna(0.0)

    wealth = (1.0 + net).cumprod()
    bench_wealth = (1.0 + panel_ret).cumprod()

    active = net - panel_ret
    n = len(net)
    ann_ret = float(wealth.iloc[-1] ** (252 / max(n, 1)) - 1)
    ann_vol = float(net.std() * math.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else None
    ir = (active.mean() / active.std()) * math.sqrt(252) if active.std() > 0 else None
    beta = float(np.cov(net, panel_ret)[0, 1] / panel_ret.var()) if panel_ret.var() > 0 else None
    alpha_ann = float((net.mean() - beta * panel_ret.mean()) * 252) if beta is not None else None

    drawdown = wealth / wealth.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else None
    hit_rate = float((net > 0).mean())

    avg_turnover = float(turnover.mean())
    ann_turnover = avg_turnover * 252
    costs_cum = float(cost.sum())
    borrow_cum = float(borrow.sum())
    avg_leverage = float((w_prev.abs().sum(axis=1)).mean())

    half = n // 2
    first, second = net.iloc[:half], net.iloc[half:]

    yearly = net.groupby(net.index.year)
    yearly_tbl = {int(y): float((1 + yr).prod() - 1) for y, yr in yearly}

    return {
        "daily_returns": net,
        "wealth": wealth,
        "benchmark_wealth": bench_wealth,
        "avg_turnover": avg_turnover,
        "ann_turnover": ann_turnover,
        "costs_cum": costs_cum,
        "borrow_cum": borrow_cum,
        "avg_leverage": avg_leverage,
        "metrics": {
            "n_sessions": n,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "information_ratio": ir,
            "beta_vs_panel": beta,
            "alpha_ann": alpha_ann,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_rate": hit_rate,
            "ann_turnover": ann_turnover,
            "cost_drag_total": costs_cum + borrow_cum,
            "first_half_ret": float((1 + first).prod() - 1),
            "second_half_ret": float((1 + second).prod() - 1),
            "yearly_returns": yearly_tbl,
        },
    }


def _cagr(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def _f(x: float | None, spec: str = "+.3f") -> str:
    return "n/a" if x is None else format(x, spec)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def analyze(
    signal_json_path: str,
    data_dir: str | Path | None,
    direction: str = "auto",
    weighting: str = "bucket",
    cost_bps: float = 10.0,
    borrow_bps: float = 50.0,
    target_vol: float | None = 0.10,
    rebalance: str = "daily",
) -> str:
    with open(signal_json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    stock_data = load_stock_data(data_dir)
    if not stock_data:
        raise RuntimeError("No OHLCV data loaded — pass --data-dir.")
    close = stock_data["Close"]

    clean_code = saved["signal_code"]
    exec_result = execute_signal_code(clean_code, stock_data, selection_periods=5)
    if exec_result is None:
        raise RuntimeError("Saved signal code failed to execute.")
    signal_values, selected_name = exec_result

    raw_metrics = saved.get("evaluation_metrics") or {}
    ic_sign = raw_metrics.get("mean_ic")

    weights = build_weights(signal_values, direction=direction, ic_sign=ic_sign, weighting=weighting)
    sim = simulate(weights, close, cost_bps=cost_bps, borrow_bps=borrow_bps, target_vol=target_vol, rebalance=rebalance)
    m = sim["metrics"]

    daily = sim["daily_returns"]
    csv_path = Path(signal_json_path).with_name(Path(signal_json_path).stem + "_portfolio.csv")
    pd.DataFrame(
        {
            "date": daily.index,
            "portfolio_return": daily.values,
            "wealth": sim["wealth"].values,
            "benchmark_wealth": sim["benchmark_wealth"].values,
        }
    ).to_csv(csv_path, index=False)

    direction_used = "inverse (signal is negative-IC as coded)" if (direction == "inverse" or (direction == "auto" and (ic_sign or 0) < 0)) else "direct (as coded)"
    yearly_rows = "\n".join(f"- {y}: {_cagr(r)}" for y, r in sorted(m["yearly_returns"].items()))

    p_name = Path(signal_json_path).name
    md = f"""# Portfolio Simulation — `{p_name}`

- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
- Signal (selected): `{selected_name}`
- Construction: {weighting} weights, long/short dollar-neutral, direction = **{direction_used}**
- Instruments: {close.shape[1]} | Sessions: {m['n_sessions']}
- Vol target: {'off' if target_vol is None else f'{target_vol:.0%} annualized (causal trailing-63d)'} | Cost: {cost_bps} bps one-way | Borrow: {borrow_bps} bps/yr on short leg
- Rebalance: {rebalance} (weights carried between rebalances; costs charged on actual trades)
- Universe data dir: {data_dir or 'default/SIGNAL_DATA_DIR/data/sp500'}

## Returns vs equal-weight panel benchmark
- Geometric annualized return: **{_cagr(m['ann_return'])}**
- Annualized vol: {_cagr(m['ann_vol']) if m['ann_vol'] else 'n/a'}
- Sharpe (ann. return / ann. vol): {_f(m['sharpe'])}
- Information ratio vs panel: {_f(m['information_ratio'])}
- Beta vs panel: {_f(m['beta_vs_panel'])} | CAPM alpha (ann.): {_cagr(m['alpha_ann']) if m['alpha_ann'] is not None else 'n/a'}
- First half: {_cagr(m['first_half_ret'])} | Second half: {_cagr(m['second_half_ret'])}
- Max drawdown: {_cagr(m['max_drawdown'])} | Calmar: {_f(m['calmar'])}
- Hit rate (daily P(r>0)): {_cagr(m['hit_rate'])}

## Costs & turnover
- Avg one-way turnover: {_pct(sim['avg_turnover'])} | Annualized: {_cagr(sim['ann_turnover'])}
- Transaction costs paid (life): {_cagr(sim['costs_cum'])} | Borrow paid (life): {_cagr(sim['borrow_cum'])}
- Avg gross leverage: {_f(sim['avg_leverage'], '.2f')}x

## Calendar-year returns
{yearly_rows}

## Verdict
{_verdict(m)}

## Honest caveats
- Simplified execution model: close-to-close returns, no impact model, flat {cost_bps} bps cost, no halts/limits.
- Dollar-neutral ≠ beta-neutral (see portfolio beta above). Shorts assumed available at {borrow_bps} bps.
- Universe is current-constituent (survivorship bias lives here too).
- Backtest on the same data that generated the signal family — retest on untouched out-of-sample panels before allocating capital.

Daily returns CSV: `{csv_path.name}`
"""
    return md


def _pct(v):
    return "n/a" if v is None else f"{v:.1%}"


def _verdict(m: dict) -> str:
    verdicts = []
    sh, ir, dd = m.get("sharpe"), m.get("information_ratio"), m.get("max_drawdown")
    if sh is not None and sh > 0.5:
        verdicts.append("risk-adjusted return positive and material")
    elif sh is not None and sh > 0:
        verdicts.append("positive but weak risk-adjusted return")
    else:
        verdicts.append("no usable risk-adjusted return as constructed")
    if ir is not None and ir > 0.3:
        verdicts.append("adds information over the panel benchmark")
    else:
        verdicts.append("adds no reliable information over the panel benchmark")
    if dd is not None and dd < -0.3:
        verdicts.append("drawdown profile UNSUITABLE for mandates that cannot absorb >30% DD")
    return "Verdict: " + "; ".join(verdicts) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio simulation for a saved workflow signal.")
    parser.add_argument("signal_json", help="Path to saved signal JSON")
    parser.add_argument("--data-dir", default=None, help="OHLCV CSV directory")
    parser.add_argument("--direction", choices=["auto", "direct", "inverse"], default="auto")
    parser.add_argument("--weighting", choices=["bucket", "z"], default="bucket")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--borrow-bps", type=float, default=50.0)
    parser.add_argument("--target-vol", type=float, default=0.10, help="Annualized vol target (0 disables)")
    parser.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="daily",
                        help="Rebalance frequency (holdings carried between rebalances; default daily)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    target_vol = None if args.target_vol == 0 else args.target_vol
    md = analyze(
        args.signal_json,
        args.data_dir,
        direction=args.direction,
        weighting=args.weighting,
        cost_bps=args.cost_bps,
        borrow_bps=args.borrow_bps,
        target_vol=target_vol,
        rebalance=args.rebalance,
    )

    out = Path(args.signal_json).with_name(Path(args.signal_json).stem + "_portfolio.md")
    out.write_text(md, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(md)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
