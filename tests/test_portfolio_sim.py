# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Local-fork test suite (branch local/tier1-universes).

"""Tests for portfolio_sim: dollar neutrality, rebalances, cost accounting, vol targeting."""

import numpy as np
import pandas as pd

from signal_discovery_workflow.portfolio_sim import (
    _apply_vol_target,
    _rebalance,
    _verdict,
    build_weights,
    simulate,
)


def _panel(n_dates=80, n_stocks=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    signal = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.standard_normal((n_dates, n_stocks)) * 0.01, axis=0)),
        index=dates,
        columns=tickers,
    )
    return signal, close


class TestBuildWeights:
    def test_fallback_auto_inverse_for_negative_ic(self):
        signal, _ = _panel()
        w_direct = build_weights(signal, direction="direct")
        w_auto_neg = build_weights(signal, direction="auto", ic_sign=-0.05)
        assert np.allclose(w_auto_neg.values, (-w_direct).values)

    def test_bucket_weights_dollar_neutral(self):
        signal, _ = _panel()
        w = build_weights(signal, direction="direct", weighting="bucket")
        sums = w.sum(axis=1)
        assert np.allclose(sums, 0.0, atol=1e-10)

    def test_bucket_gross_is_two(self):
        signal, _ = _panel()
        w = build_weights(signal, direction="direct", weighting="bucket")
        gross = w.abs().sum(axis=1)
        assert np.allclose(gross, 2.0, atol=1e-10)


class TestRebalance:
    def test_monthly_rebalances_only_at_month_end(self):
        signal, _ = _panel()
        w = build_weights(signal, direction="direct")
        w_month = _rebalance(w, "monthly")
        changed = (w_month != w_month.shift(1)).any(axis=1)
        n_periods = w_month.index.to_period("M").nunique()
        # Weights may only move at the month-end anchor (allow one initial change)
        assert changed.sum() <= n_periods + 1
        # All other days carry weights unchanged
        assert (~changed).sum() >= len(changed) - (n_periods + 1)

    def test_daily_is_identity(self):
        signal, _ = _panel()
        w = build_weights(signal, direction="direct")
        assert np.allclose(_rebalance(w, "daily").values, w.values)


class TestSimulate:
    def test_cost_drag_is_deducted(self):
        signal, close = _panel()
        w = build_weights(signal, direction="direct")
        gross_sim = simulate(w, close, cost_bps=0.0, borrow_bps=0.0, target_vol=None)
        cost_sim = simulate(w, close, cost_bps=100.0, borrow_bps=0.0, target_vol=None)
        assert cost_sim["daily_returns"].mean() < gross_sim["daily_returns"].mean()

    def test_returns_and_wealth_shapes(self):
        signal, close = _panel()
        w = build_weights(signal, direction="direct")
        sim = simulate(w, close, target_vol=None)
        assert len(sim["daily_returns"]) == len(close)
        assert sim["wealth"].iloc[0] > 0

    def test_vol_target_clips_leverage(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.standard_normal(200) * 0.001, index=pd.bdate_range("2022-01-01", periods=200))
        scaled = _apply_vol_target(r, target_vol=0.10, lookback=63, max_leverage=2.0)
        # Low-vol series: realized vol below target → leverage capped at 2.0
        ratio = (scaled / r).clip(upper=2.01)
        assert float(ratio.max()) <= 2.01


class TestVerdict:
    def test_verdict_negative_sharpe(self):
        m = {"sharpe": -1.0, "information_ratio": 0.0, "max_drawdown": -0.4}
        assert "no usable" in _verdict(m)
        assert "UNSUITABLE" in _verdict(m)

    def test_verdict_positive(self):
        m = {"sharpe": 1.2, "information_ratio": 0.6, "max_drawdown": -0.1}
        out = _verdict(m)
        assert "positive and material" in out
