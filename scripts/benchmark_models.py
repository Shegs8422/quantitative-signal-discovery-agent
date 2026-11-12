"""
Model benchmark for the Signal Discovery Agent (local fork tool).

Runs the *same* HAC-gated config against several Nemotron models and reports
signal quality vs. token cost — the "cost per accepted signal" question.

Reads models available on integrate.api.nvidia.com (no guessing), generates
configs from configs/config-optimization-hac.yml, runs `nat run` via a Python
subprocess (quoting-safe), parses each run's JSON result, and aggregates.

Usage:
    python scripts/benchmark_models.py                  # 2 runs/model, momentum signals
    python scripts/benchmark_models.py --runs 1 --request "volatility signals"

Writes docs/BENCHMARK.md. Uses NVIDIA_API_KEY from .env (or the environment).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "config-optimization-hac.yml"
BENCH_DIR = ROOT / ".benchmarks"
OUT_MD = ROOT / "docs" / "BENCHMARK.md"

# Only models this account is entitled to are benchmarked (catalog visibility !=
# access — everything else 404s on the hosted function; verified by direct API call).
MODELS = [
    ("lightning-30b", "nvidia/nemotron-3.5-lightning-30b-a3b"),   # deployed default
    ("nano-omni-30b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),  # 3.x omni reasoning
    ("bolt-e4b", "nvidia/nemotron-3.5-bolt-e4b"),                  # 3.5 smallest/fastest
]


def _api_key() -> str:
    env = os.environ.get("NVIDIA_API_KEY")
    if env:
        return env.strip()
    env_file = ROOT / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("NVIDIA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("NVIDIA_API_KEY not found in environment or .env")


def check_catalog(models: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Verify each model id exists in the hosted catalog (free metadata call)."""
    import urllib.request

    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/models",
        headers={"Authorization": f"Bearer {_api_key()}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        catalog = {m["id"] for m in json.load(r).get("data", [])}
    ok, missing = [], []
    for label, model_id in models:
        (ok if model_id in catalog else missing).append(model_id)
    if missing:
        print(f"WARNING: not in catalog, skipping: {', '.join(missing)}")
    return [(label, model_id) for label, model_id in models if model_id in catalog]


def make_config(label: str, model_id: str) -> Path:
    """Config identical to config-optimization-hac.yml, different model + Phoenix project."""
    BENCH_DIR.mkdir(exist_ok=True)
    text = BASE_CONFIG.read_text(encoding="utf-8")
    text = text.replace("nvidia/nemotron-3.5-lightning-30b-a3b", model_id)
    text = text.replace("project: signal-discovery-workflow-hac", f"project: benchmark-{label}")
    out = BENCH_DIR / f"config-{label}.yml"
    out.write_text(text, encoding="utf-8")
    return out


def run_once(config_path: Path, request: str, label: str, run_index: int) -> dict:
    """Run nat run; return (parsed_result | None, wall_seconds, raw_tail)."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["uv", "run", "--env-file", ".env", "nat", "run",
         "--config_file", str(config_path), "--input", request],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall = time.perf_counter() - t0
    out = proc.stdout
    log_dir = BENCH_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{label}_run{run_index}.log").write_text(out, encoding="utf-8", errors="replace")
    result = None
    for m in list(re.finditer(r'\{\s*"status"\s*:', out))[::-1]:
        try:
            obj, _ = json.JSONDecoder(strict=False).raw_decode(out[m.start():])
            if isinstance(obj, dict) and obj.get("status"):
                result = obj
                break
        except Exception:
            continue
    return {"result": result, "wall_s": round(wall, 1), "exit": proc.returncode}


def phoenix_token_totals() -> dict[str, dict[str, int]]:
    """Per-project LLM token sums (this run's projects only)."""
    import sqlite3

    db = Path.home() / ".phoenix" / "phoenix.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.text_factory = str
    rows = con.execute(
        """
        SELECT s.name, SUM(s.llm_token_count_prompt), SUM(s.llm_token_count_completion), COUNT(*)
        FROM spans s
        WHERE s.span_kind = 'LLM'
        GROUP BY s.name
        """
    ).fetchall()
    return {r[0]: {"prompt_tokens": int(r[1] or 0), "completion_tokens": int(r[2] or 0), "calls": int(r[3])} for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--request", default="momentum signals")
    parser.add_argument("--only", default=None, help="Benchmark only this model label (e.g. nano-omni-30b)")
    args = parser.parse_args()

    os.chdir(ROOT)
    models = check_catalog(MODELS)
    if args.only:
        models = [m for m in models if m[0] == args.only]
    if not models:
        raise RuntimeError("No benchmark models found in the hosted catalog.")

    records = []
    for label, model_id in models:
        cfg = make_config(label, model_id)
        for i in range(1, args.runs + 1):
            print(f"\n=== {label} run {i}/{args.runs} — {args.request} ===")
            r = run_once(cfg, args.request, label, i)
            res = r["result"] or {}
            m = res.get("metrics", {})
            ic = m.get("mean_ic")
            rec = {
                "model": label,
                "model_id": model_id,
                "run": i,
                "request": args.request,
                "status": res.get("status"),
                "iteration": res.get("iteration"),
                "mean_ic": ic,
                "abs_ic": abs(ic) if isinstance(ic, (int, float)) else None,
                "hac_p_value": m.get("hac_p_value"),
                "avg_turnover": m.get("avg_turnover"),
                "recommended_direction": res.get("recommended_direction"),
                "wall_s": r["wall_s"],
                "signals_count": len(res.get("signals", [])),
            }
            records.append(rec)
            print(f"-> status={rec['status']} meanIC={rec['mean_ic']} HAC p={rec['hac_p_value']} wall={rec['wall_s']}s")

    tokens = phoenix_token_totals()

    totals: dict[str, dict] = {}
    for label, _ in models:
        rs = [r for r in records if r["model"] == label]
        accepted = sum(1 for r in rs if r["status"] == "accepted")
        ic_vals = [r["abs_ic"] for r in rs if r["abs_ic"] is not None]
        totals[label] = {
            "runs": len(rs), "accepted": accepted,
            "accept_rate": accepted / len(rs) if rs else 0.0,
            "mean_abs_ic": sum(ic_vals) / len(ic_vals) if ic_vals else None,
            "avg_wall_s": sum(r["wall_s"] for r in rs) / len(rs) if rs else None,
        }

    lines = [
        "# Benchmark — cost per accepted signal (Nemotron family)",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S} | Request: `{args.request}` | Runs/model: {args.runs} | Config: `configs/config-optimization-hac.yml` (identical for all models; only `model_name` + Phoenix project differ)",
        "",
        "| Model | id | Status count | Avg wall | mean |IC| | HAC p (last) | Tokens (all LLM calls) | Accept rate | Cost per accepted (tokens) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, model_id in models:
        t = totals[label]
        mtok = tokens.get(model_id) or {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        tok_total = mtok["prompt_tokens"] + mtok["completion_tokens"]
        per_accept = round(tok_total / t["accepted"]) if t["accepted"] else "n/a (0 accepted)"
        lines.append(
            f"| `{label}` | `{model_id}` | {t['accepted']}/{t['runs']} accepted "
            f"| {_f(t['avg_wall_s'])}s | {_f(t['mean_abs_ic'])} | see runs | {tok_total:,} | {t['accept_rate']:.0%} | {per_accept if isinstance(per_accept, str) else f'{per_accept:,}'} |"
        )
    lines += [
        "",
        "## Per-run detail",
        "",
        "| Model | Run | Status | mean IC | HAC p | turnover | direction | wall (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['model']} | {r['run']} | {r['status']} | {_f(r['mean_ic'])} "
            f"| {_f(r['hac_p_value'])} | {_f(r['avg_turnover'])} | {r['recommended_direction']} | {r['wall_s']} |"
        )
    lines += [
        "",
        "## Reading the table",
        "",
        "- Token costs are reported without invented prices — plugin the $/token for your endpoint and",
        "  divide `cost per accepted (tokens)` by it to get cost-per-acceptance.",
        "- `n/a` means the model produced no acceptance across the sampled runs.",
        "- Same seed budget & prompts across models; differences reflect the model, not the harness.",
        "- Token totals are CUMULATIVE per LLM span name across this machine's Phoenix history (indicated).",
        "",
        "## Caveats",
        "",
        "- 2 runs per model is a directional sample, not a significance test of model quality.",
        "- Acceptance here means clearing the HAC-gated config at 0.02 |IC| — not deployability.",
        "- Wall-clock is dominated by the slowest LLM call; interpret with care across endpoints.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nBenchmark written to {OUT_MD}")


def _f(v):
    return "n/a" if v is None else f"{v:+.4f}"


if __name__ == "__main__":
    main()
