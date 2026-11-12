#!/bin/sh
# Offline 60-second demo — no API key, no network.
# Proves the assurance pipeline end-to-end on the committed synthetic example.
set -eu

echo "1/4 Installing into .venv (uv sync --locked)..."
uv sync --locked

echo "2/4 Running test suite..."
uv run pytest tests/ | tail -n 3

echo "3/4 Accuracy report on the synthetic demo signal..."
uv run python -m signal_discovery_workflow.signal_report examples/demo_signal.json --data-dir examples/data --skip-expensive

echo "4/4 Portfolio simulation (monthly rebalance) on the same signal..."
uv run python -m signal_discovery_workflow.portfolio_sim examples/demo_signal.json --data-dir examples/data --rebalance monthly

echo ""
echo "Demo complete. See examples/*_report.md and examples/*_portfolio.md."
echo "To try the live LLM agents you need NVIDIA_API_KEY in .env (see README)."
