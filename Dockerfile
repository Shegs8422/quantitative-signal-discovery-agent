# Quantitative Signal Discovery Agent — enhanced fork (research/engineering demo)
#
# Build:  docker build -t qsda .
# Run:    docker run --rm qsda            # offline demo: report + portfolio sim (no API key)
# Tests:  docker run --rm qsda pytest     # pytest suite (115 tests)
# Real workflow (needs key): docker run --rm -e NVIDIA_API_KEY=nvapi-... qsda nat run --config_file configs/config-optimization-hac.yml --input "momentum signals"

FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir uv

ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra test --no-install-project

COPY . .
RUN uv sync --locked --extra test

ENV PATH="/app/.venv/bin:$PATH"

CMD ["sh", "-c", "python -m signal_discovery_workflow.signal_report examples/demo_signal.json --data-dir examples/data --skip-expensive && python -m signal_discovery_workflow.portfolio_sim examples/demo_signal.json --data-dir examples/data --rebalance monthly"]
