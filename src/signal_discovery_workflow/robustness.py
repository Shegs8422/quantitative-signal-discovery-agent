# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Local-fork addition (branch feat/tier1-universes-assurance). Not part of the upstream example.

"""
Robustness frontier (local fork): how does a saved signal behave across
direction × cost × rebalance-frequency? A fast way to see whether a "winner"
is a robust edge or a one-tweak artifact — before anyone discusses capital.

Usage:
    python -m signal_discovery_workflow.robustness examples/demo_signal.json --data-dir examples/data
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .portfolio_sim import build_weights, simulate
from .signal_evaluator import execute_signal_code, load_stock_data

logger = logging.getLogger(__name__)

GRID_COSTS = [2.0, 10.0, 50.0]
GRID_REBAL = ["daily", "weekly", "monthly"]
GRID_DIRECTIONS = ["direct", "inverse"]


def analyze(signal_json_path: str, data_dir: str | Path | None,
            target_vol: float | None = 0.10, borrow_bps: float = 50.0) -> str:
    with open(signal_json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    stock_data = load_stock_data(data_dir)
    if not stock_data:
        raise RuntimeError("No OHLCV data loaded — pass --data-dir.")
    close = stock_data["Close"]

    exec_result = execute_signal_code(saved["signal_code"], stock_data, selection_periods=5)
    if exec_result is None:
        raise RuntimeError("Saved signal code failed to execute.")
    signal_values, selected_name = exec_result

    raw_metrics = saved.get("evaluation_metrics") or {}
    ic_sign = raw_metrics.get("mean_ic")
    p_name = Path(signal_json_path).name

    rows = []
    for direction in GRID_DIRECTIONS:
        weights = build_weights(signal_values, direction=direction, ic_sign=ic_sign, weighting="bucket")
        for cost in GRID_COSTS:
            for freq in GRID_REBAL:
                sim = simulate(weights, close, cost_bps=cost, borrow_bps=borrow_bps,
                               target_vol=target_vol, rebalance=freq)
                m = sim["metrics"]
                rows.append({
                    "direction": direction, "cost_bps": cost, "rebalance": freq,
                    "ann_return": m["ann_return"], "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"], "ann_vol": m["ann_vol"],
                    "turnover": m["ann_turnover"], "alpha_ann": m["alpha_ann"],
                })
                logger.info(f"{direction:7s} {cost:>4.0f}bps {freq:7s} → Sharpe {m['sharpe']:+.2f}")

    n = len(rows)
    n_positive_sharpe = sum(1 for r in rows if r["sharpe"] and r["sharpe"] > 0)
    n_cost_breakeven = sum(1 for r in rows if r["ann_return"] and r["ann_return"] > 0)

    table_lines = [
        "| Direction | Cost (bps) | Rebalance | Ann. return | Sharpe | Ann. vol | Max DD | Turnover/yr | Alpha (ann.) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        table_lines.append(
            f"| {r['direction']:<9} | {r['cost_bps']:g} | {r['rebalance']:<7} "
            f"| {r['ann_return']:+.2%} | {r['sharpe']:+.2f} | {r['ann_vol']:+.2%} "
            f"| {r['max_drawdown']:+.1%} | {r['turnover']:+.0f} | {_c(r['alpha_ann'])} |"
        )

    md = f"""# Robustness Frontier — `{p_name}`

- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
- Signal: `{selected_name}` | Instruments: {close.shape[1]} | Sessions: {close.shape[0]}
- Direction × cost grid × rebalance frequency. All cells use the same saved module.

{chr(10).join(table_lines)}

## Reading the frontier
- Cells across **direction** reveal sign stability: a real edge survives the sign flip; a fake one flips the outcome.
- Cells across **cost** reveal cost sensitivity: an edge with a Sharpe cliff between 2 and 10 bps is not tradable.
- Cells across **rebalance** reveal turnover intensity: momentum signals usually improve materially on slower rebalances.
- **{n_positive_sharpe}/{n}** cells have positive Sharpe; **{n_cost_breakeven}/{n}** are net-positive after costs.

> Honest use: report the whole frontier, never the best cell. Picking the best cell off this grid is itself a
> multiple-testing event and must be counted in the trial registry.
"""
    return md


def _c(v):
    return "n/a" if v is None else f"{v:+.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness frontier for a saved workflow signal.")
    parser.add_argument("signal_json", help="Path to the saved signal JSON")
    parser.add_argument("--data-dir", default=None, help="OHLCV CSV directory")
    parser.add_argument("--target-vol", type=float, default=0.10, help="Annualized vol target (0 disables)")
    parser.add_argument("--borrow-bps", type=float, default=50.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    target_vol = None if args.target_vol == 0 else args.target_vol
    md = analyze(args.signal_json, args.data_dir, target_vol=target_vol, borrow_bps=args.borrow_bps)

    out = Path(args.signal_json).with_name(Path(args.signal_json).stem + "_robustness.md")
    out.write_text(md, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(md)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
