# Offline 60-second demo — no API key, no network (Windows)
# Proves the assurance pipeline end-to-end on the committed synthetic example.
$ErrorActionPreference = "Stop"

Write-Host "1/4 Installing into .venv (uv sync --locked)..." -ForegroundColor Cyan
uv sync --locked

Write-Host "2/4 Running test suite..." -ForegroundColor Cyan
uv run pytest tests/ | Select-Object -Last 3

Write-Host "3/4 Accuracy report on the synthetic demo signal..." -ForegroundColor Cyan
uv run python -m signal_discovery_workflow.signal_report examples/demo_signal.json --data-dir examples/data --skip-expensive

Write-Host "4/4 Portfolio simulation (monthly rebalance) on the same signal..." -ForegroundColor Cyan
uv run python -m signal_discovery_workflow.portfolio_sim examples/demo_signal.json --data-dir examples/data --rebalance monthly

Write-Host "`nDemo complete. See examples/*_report.md and examples/*_portfolio.md." -ForegroundColor Green
Write-Host "To try the live LLM agents you need NVIDIA_API_KEY in .env (see README)." -ForegroundColor Yellow
