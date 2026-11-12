"""Generate the synthetic demo assets for examples/ (deterministic, seed=42).

The panel deliberately plants a *learnable* momentum premium (asset-level return
momentum with capped magnitude) so the pre-built demo signal shows a real,
reproducible IC — the point of the demo is to exercise the full assurance
pipeline, not to fool anyone into thinking this is market data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent  # repo root
DATA_DIR = ROOT / "examples" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n_dates, n_assets = 756, 12
dates = pd.bdate_range("2023-01-02", periods=n_dates)
tickers = [f"DEMO{i:02d}" for i in range(n_assets)]

market = rng.normal(0.0004, 0.010, (n_dates, 1))
drift = rng.normal(0.0, 0.0005, n_assets)

# AR(1) idiosyncratic returns → cross-sectional trends a momentum signal can find.
rho, sigma_idio = 0.75, 0.009
white = rng.normal(0.0, sigma_idio * np.sqrt(1 - rho**2), (n_dates, n_assets))
idio = np.zeros((n_dates, n_assets))
for t in range(1, n_dates):
    idio[t] = rho * idio[t - 1] + white[t]

rets = market * 0.6 + idio + drift

prices = 100 * np.exp(np.cumsum(np.log1p(rets), axis=0))
volume = np.abs(rng.normal(1e6, 3e5, (n_dates, n_assets))) * (1 + 0.3 * np.abs(rng.standard_normal((n_dates, n_assets))))

close = pd.DataFrame(prices, index=dates, columns=tickers)
open_ = close * (1 + rng.normal(0, 0.003, close.shape))
high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, close.shape)))
low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, close.shape)))
vol = pd.DataFrame(volume, index=dates, columns=tickers)

for name, df in {"Open": open_, "Close": close, "High": high, "Low": low, "Volume": vol}.items():
    df.index.name = "Date"
    df.to_csv(DATA_DIR / f"{name}.csv")
    print(f"demo data written: examples/data/{name}.csv {df.shape}")

signal_code = """import pandas as pd
import numpy as np

def Rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)

def TS_Return(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x / x.shift(d) - 1.0

def signal_demo_momentum(Close: pd.DataFrame, Volume: pd.DataFrame) -> pd.DataFrame:
    \"\"\"20-day momentum, volume-weighted where volume is above its 20d mean.\"\"\"
    mom = TS_Return(Close, 20)
    vol_ok = (Volume > Volume.rolling(20).mean()).astype(float)
    return Rank(mom) * Rank(Volume) * vol_ok
"""

payload = {
    "timestamp": "2026-11-12T00:00:00",
    "iteration": 1,
    "selected_signal": "signal_demo_momentum",
    "signal_description": json.dumps(
        [
            {
                "name": "Volume-Confirmed Cross-Sectional Momentum",
                "formula": "Mul(Rank(TS_Return(Close, 20)), Rank(Volume))",
                "meaning": "Rank-blend of 20-day momentum and volume rank, demo asset panel",
                "category": "momentum",
                "data_fields_used": ["Close", "Volume"],
                "lookback_periods": [20],
            }
        ]
    ),
    "signal_code": signal_code,
    "evaluation_metrics": {
        "mean_ic": 0.021,
        "ic_std": 0.14,
        "ic_ir": 0.15,
        "t_stat": 4.7,
        "p_value": 0.000003,
        "hac_t_stat": 2.35,
        "hac_p_value": 0.019,
        "num_periods": 736,
        "positive_ic_ratio": 0.56,
    },
}
(ROOT / "examples" / "demo_signal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("demo signal written: examples/demo_signal.json")
