# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Local-fork test suite (branch local/tier1-universes).

"""Tests for the local fork extensions: aliases, HAC stats, direction reporting, report tooling."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from signal_discovery_workflow.llm_utils import normalize_operator_names
from signal_discovery_workflow.signal_discovery_optimization_workflow import (
    SignalOptimizerConfig,
    _format_workflow_result,
)
from signal_discovery_workflow.signal_evaluator import compute_forward_returns, compute_rank_ic
from signal_discovery_workflow.signal_report import (
    _block_bootstrap_ci,
    _decile_long_short,
    _newey_west_stats,
)


class TestOperatorAliases:
    def test_known_hallucinations_map_to_catalogue(self):
        ops = {"TS_Vol", "TS_Kurt", "TS_Skew", "TS_Std", "EMA", "TS_Mean"}
        assert normalize_operator_names("TS_Volatility(Close, 10)", ops) == "TS_Vol(Close, 10)"
        assert normalize_operator_names("TS_Kurtosis(Close, 30)", ops) == "TS_Kurt(Close, 30)"
        assert normalize_operator_names("EWM_Mean(Close, 5)", ops) == "EMA(Close, 5)"
        assert normalize_operator_names("TS_StdDev(Close, 20)", ops) == "TS_Std(Close, 20)"

    def test_unmapped_names_are_left_alone(self):
        ops = {"TS_Vol", "EMA"}
        assert normalize_operator_names("TS_VolumeWeight(Close, 5)", ops) == "TS_VolumeWeight(Close, 5)"


class TestHACStats:
    def _aligned(self, n_dates=120, n_stocks=20, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        tickers = [f"S{i:02d}" for i in range(n_stocks)]
        returns = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
        return returns

    def test_compute_rank_ic_emits_hac_keys(self):
        returns = self._aligned()
        fwd = compute_forward_returns(returns + 1.0, periods=5)
        result = compute_rank_ic(returns, fwd)
        assert "hac_t_stat" in result and "hac_p_value" in result
        assert result["hac_p_value"] is None or 0.0 <= result["hac_p_value"] <= 1.0

    def test_newey_west_iid_detection(self):
        rng = np.random.default_rng(123)
        x = rng.standard_normal(300) + 0.25  # strong iid signal
        mean, t_stat, p_value = _newey_west_stats(x, lag=5)
        assert abs(t_stat) > 3.0
        assert p_value < 0.01

    def test_block_bootstrap_ci_contains_mean(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(200)
        ci = _block_bootstrap_ci(x, n_iter=200, block=10, seed=42)
        assert ci is not None
        lo, hi = ci
        assert lo < float(np.mean(x)) < hi


class TestDirectionReporting:
    def _config(self):
        return SimpleNamespace(ic_threshold=0.02, p_value_threshold=0.05)

    def test_negative_ic_flags_inverse_direction(self):
        result = _format_workflow_result(
            status="best_effort",
            request="momentum signals",
            iteration=1,
            total_iterations=3,
            signal_json="[]",
            ic_results={"mean_ic": -0.05, "selected_signal": "signal_x"},
            saved_path=None,
            config=self._config(),
        )
        payload = json.loads(result)
        assert payload["recommended_direction"] == -1
        assert "inverse" in payload.get("direction_note", "")

    def test_positive_ic_flags_long_direction(self):
        result = _format_workflow_result(
            status="accepted",
            request="momentum signals",
            iteration=1,
            total_iterations=3,
            signal_json="[]",
            ic_results={"mean_ic": 0.05, "selected_signal": "signal_x"},
            saved_path=None,
            config=self._config(),
        )
        payload = json.loads(result)
        assert payload["recommended_direction"] == 1
        assert "direction_note" not in payload


class TestConfigBackcompat:
    def test_fork_fields_default_to_upstream_behavior(self):
        cfg = SignalOptimizerConfig()
        assert cfg.require_hac_pvalue is False
        assert cfg.hac_pvalue_threshold == 0.05
        assert cfg.cost_bps == 0.0


class TestDecileLongShort:
    def _panel(self, n_dates=120, n_stocks=40, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2021-01-01", periods=n_dates, freq="B")
        tickers = [f"T{i:02d}" for i in range(n_stocks)]
        signal = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
        returns = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)) * 0.01, index=dates, columns=tickers)
        return signal, returns

    def test_backtest_columns_and_turnover_bounds(self):
        signal, returns = self._panel()
        ls = _decile_long_short(signal, returns, cost_bps=10.0)
        assert list(ls.columns) == ["spread", "turnover", "net_spread"]
        assert len(ls) > 50
        assert ls["turnover"].between(0.0, 1.0).all()

    def test_net_spread_is_spread_minus_cost(self):
        signal, returns = self._panel()
        ls = _decile_long_short(signal, returns, cost_bps=10.0)
        expected = ls["spread"] - (10.0 / 1e4) * ls["turnover"]
        assert np.allclose(ls["net_spread"], expected, atol=1e-12)
