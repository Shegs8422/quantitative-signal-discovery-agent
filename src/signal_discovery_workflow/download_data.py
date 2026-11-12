# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Disclaimer:
# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.

"""
Download S&P 500 price-volume data using yfinance.

Fetches Open, Close, High, Low, and Volume data for a curated S&P 500
ticker universe and saves each field as a separate CSV file compatible
with the signal discovery workflow.

Usage:
    python -m signal_discovery_workflow.download_data
    python -m signal_discovery_workflow.download_data --start 2015-01-01 --end 2025-12-31
    python -m signal_discovery_workflow.download_data --output /path/to/output
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_FIELDS = ["Open", "Close", "High", "Low", "Volume"]

DATA_ROOT = Path(__file__).parent / "data"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "sp500"


SP500_TICKERS = [
    'A', 'AAPL', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG',
    'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMT', 'AMZN', 'AON', 'AOS', 'APA',
    'APD', 'APH', 'ARE', 'ATO', 'AVB', 'AVY', 'AXON', 'AXP', 'AZO',
    'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK',
    'BMY', 'BRO', 'BSX', 'BWA', 'BXP',
    'C', 'CAG', 'CAH', 'CAT', 'CB', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
    'CMCSA', 'CME', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL',
    'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CVS', 'CVX',
    'D', 'DD', 'DE', 'DECK', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DPZ', 'DRI', 'DTE', 'DUK',
    'DVA', 'DVN',
    'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES',
    'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXR',
    'F', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FRT',
    'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',
    'HAL', 'HAS', 'HBAN', 'HD', 'HIG', 'HOLX', 'HON', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM',
    'IBM', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
    'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM',
    'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR',
    'L', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LUV', 'LVS',
    'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM',
    'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPWR', 'MRK', 'MS', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
    'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',
    'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OXY',
    'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PNC', 'PNR',
    'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PTC', 'PWR', 'QCOM',
    'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY',
    'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
    'STX', 'STZ', 'SWK', 'SWKS', 'SYK', 'SYY',
    'T', 'TAP', 'TDY', 'TECH', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO',
    'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL',
    'UDR', 'UHS', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
    'VLO', 'VMC', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ',
    'WAB', 'WAT', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN',
    'XEL', 'XOM', 'YUM', 'ZBH', 'ZBRA',
]


# ---------------------------------------------------------------------------
# Universe presets (Tier 1: index/ETF panels).
#
# Every preset is a compact, stable panel of liquid instruments tradable via
# yfinance with full OHLCV history (mostly since 2012). Lists are snapshots as
# of late 2026 — refresh them periodically; the number of names (>=10) only
# needs to exceed the cross-sectional IC minimum-observations threshold.
# ---------------------------------------------------------------------------

# Curated snapshot of Nasdaq-100 constituents (mega/large-cap tech & growth).
NASDAQ100_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'AMD', 'ADBE', 'CSCO', 'PEP', 'QCOM', 'INTC', 'TMUS', 'INTU', 'AMGN',
    'TXN', 'HON', 'AMAT', 'ISRG', 'BKNG', 'ADI', 'VRTX', 'ADP', 'GILD', 'REGN',
    'LRCX', 'MU', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'MDLZ', 'CME', 'PYPL', 'MELI',
    'MAR', 'ADSK', 'CPRT', 'MRVL', 'CRWD', 'FTNT', 'ABNB', 'ROST', 'CTAS', 'AEP',
    'PCAR', 'ORLY', 'LULU', 'PAYX', 'MCHP', 'KDP', 'AXON', 'ROP', 'MNST', 'DXCM',
    'FANG', 'DASH', 'BKR', 'APP', 'TTD', 'VRSN', 'IDXX', 'EXC', 'CCEP', 'CMCSA',
    'CSX', 'CTSH', 'EA', 'SMCI', 'GFS', 'TEAM', 'ZS', 'DDOG', 'WDAY', 'FAST',
    'PDD', 'JD', 'BIDU', 'NTES', 'BILI', 'MDB', 'OKTA', 'ZM', 'DOCU', 'ILMN',
    'ALGN', 'KHC', 'CHTR', 'AMCR', 'ODFL', 'XEL',
]

# Price-weighted Dow Jones Industrial Average (30 names).
DOW30_TICKERS = [
    'MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO',
    'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT',
    'NKE', 'NVDA', 'CRM', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT',
]

# MSCI single-country equity ETFs — a clean "indices as instruments" panel
# (a single ^index cannot produce a cross-sectional Rank IC on its own).
COUNTRY_ETF_TICKERS = [
    'EWA',  # Australia
    'EWC',  # Canada
    'EWD',  # Sweden
    'EWG',  # Germany
    'EWH',  # Hong Kong
    'EWI',  # Italy
    'EWJ',  # Japan
    'EWL',  # Switzerland
    'EWN',  # Netherlands
    'EWP',  # Spain
    'EWQ',  # France
    'EWY',  # South Korea
    'EWZ',  # Brazil
    'EWT',  # Taiwan
    'THD',  # Thailand
    'INDA', # India (2012 inception)
    'FXI',  # China large-cap
    'MCHI', # MSCI China (2011 inception)
    'EPOL', # Poland (2010 inception)
    'EIDO', # Indonesia (2010 inception)
]

# SPDR S&P 500 sector ETFs. XLRE (2015) and XLC (2018) have shorter history;
# all other sectors date back to 1998. 10 names — Rank IC is noisy but valid.
SECTOR_ETF_TICKERS = [
    'XLB', 'XLE', 'XLF', 'XLI', 'XLK',
    'XLP', 'XLU', 'XLV', 'XLY', 'XLRE',
]

# Fixed income / rates ETFs — duration, credit, inflation, and aggregate exposure.
BOND_ETF_TICKERS = [
    'TLT',  # 20+ year Treasuries
    'IEF',  # 7-10 year Treasuries
    'SHY',  # 1-3 year Treasuries
    'AGG',  # US Aggregate Bond
    'TIP',  # TIPS
    'LQD',  # Investment-grade credit
    'HYG',  # High yield (2007)
    'EMB',  # EM USD sovereign (2007)
    'BND',  # Total bond market (2007)
    'GOVT', # Treasuries blended (2006)
    'MUB',  # Muni bonds (2007)
    'JNK',  # High yield (2006)
]

# Broad global bond & credit ETF panel (P2.11): duration, credit, inflation,
# intl hedged/unhedged aggregate, muni and linkers. ~30 names, all with history
# dating to ~2007+. Note: hedged intl funds (BNDX, IAGG) carry no FX exposure
# by construction; unhedged country panels DO (see --local-currency in signal_report).
GLOBAL_BOND_ETF_TICKERS = [
    'TLT', 'IEF', 'SHY', 'AGG', 'BND', 'GOVT', 'TIP', 'STIP', 'SCHP',
    'LQD', 'VCIT', 'MBB', 'MUB', 'BIV', 'FLOT', 'SRLN', 'BKLN',
    'HYG', 'JNK', 'HYLB', 'FALN', 'ANGL', 'HYDB',
    'EMB', 'EMLC', 'CWB', 'IAGG', 'BNDX',
]

# Factor ETFs — momentum, quality, value, min-vol, size, blend styles.
FACTOR_ETF_TICKERS = [
    'MTUM', # Momentum (2013)
    'QUAL', # Quality (2013)
    'VLUE', # Value (2013)
    'USMV', # Min volatility (2011)
    'SIZE', # Size (2013)
    'IWD',  # Russell 1000 Value
    'IWF',  # Russell 1000 Growth
    'RSP',  # S&P 500 equal weight (2003)
    'IWM',  # Russell 2000
    'IWN',  # Russell 2000 Value
]

UNIVERSES: dict[str, list[str]] = {
    "sp500": SP500_TICKERS,
    "nasdaq100": NASDAQ100_TICKERS,
    "dow30": DOW30_TICKERS,
    "country_etfs": COUNTRY_ETF_TICKERS,
    "sector_etfs": SECTOR_ETF_TICKERS,
    "bond_etfs": BOND_ETF_TICKERS,
    "global_bond_etfs": GLOBAL_BOND_ETF_TICKERS,
    "factor_etfs": FACTOR_ETF_TICKERS,
}

# FX pairs used to decompose country-ETF USD returns into local-currency and
# FX components. Values are yfinance symbols; invert=True means the pair is
# quoted as CCY per USD (multiply USD→CCY legs by 1/pair to get USD per CCY).
FX_PAIRS: dict[str, dict] = {
    "AUDUSD=X": {"fallbacks": ["AUD=X"], "invert": False},
    "USDCAD=X": {"fallbacks": ["CAD=X"], "invert": True},
    "USDSEK=X": {"fallbacks": ["SEK=X"], "invert": True},
    "EURUSD=X": {"fallbacks": ["EUR=X"], "invert": False},
    "USDHKD=X": {"fallbacks": ["HKD=X"], "invert": True},
    "USDJPY=X": {"fallbacks": ["JPY=X"], "invert": True},
    "USDCHF=X": {"fallbacks": ["CHF=X"], "invert": True},
    "USDKRW=X": {"fallbacks": ["KRW=X"], "invert": True},
    "USDBRL=X": {"fallbacks": ["BRL=X"], "invert": True},
    "USDTWD=X": {"fallbacks": ["TWD=X"], "invert": True},
    "USDTHB=X": {"fallbacks": ["THB=X"], "invert": True},
    "USDINR=X": {"fallbacks": ["INR=X"], "invert": True},
    "USDCNY=X": {"fallbacks": ["CNY=X"], "invert": True},
    "USDPLN=X": {"fallbacks": ["PLN=X"], "invert": True},
    "USDIDR=X": {"fallbacks": ["IDR=X"], "invert": True},
}


def get_sp500_tickers() -> list[str]:
    """Return the configured S&P 500 ticker universe."""
    logger.info(f"Using {len(SP500_TICKERS)} S&P 500 tickers")
    return SP500_TICKERS


def get_universe_tickers(universe: str) -> list[str]:
    """Return the deduplicated ticker list for a named universe preset."""
    tickers = UNIVERSES.get(universe)
    if tickers is None:
        valid = ", ".join(sorted(UNIVERSES))
        raise ValueError(f"Unknown universe '{universe}'. Valid options: {valid}")
    tickers = list(dict.fromkeys(tickers))
    logger.info(f"Using universe '{universe}' with {len(tickers)} tickers")
    return tickers


def download_universe_data(
    universe: str = "sp500",
    start: str = "2012-01-01",
    end: str = "2025-12-31",
    output_dir: str | Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a named universe preset (or an explicit ticker list).

    Args:
        universe: Name of a preset from ``UNIVERSES`` (e.g. "sp500", "nasdaq100",
            "dow30", "country_etfs", "sector_etfs", "bond_etfs", "factor_etfs").
            Ignored for ticker selection when ``tickers`` is provided.
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.
        output_dir: Directory to save CSV files. Defaults to ``data/<universe>/``
            (or ``data/custom/`` when an explicit ticker list is passed).
        tickers: Explicit ticker list overriding the universe preset.

    Returns:
        Dictionary mapping field names to DataFrames.
    """
    if tickers is not None:
        ticker_list = list(dict.fromkeys(t.strip() for t in tickers if t.strip()))
        universe_key = "custom"
    else:
        ticker_list = get_universe_tickers(universe)
        universe_key = universe

    if output_dir:
        output_path = Path(output_dir)
    elif universe_key == "sp500":
        output_path = DEFAULT_OUTPUT_DIR  # backwards-compatible default location
    else:
        output_path = DATA_ROOT / universe_key
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading data from {start} to {end} for {len(ticker_list)} tickers...")
    raw = yf.download(ticker_list, start=start, end=end, group_by="column", auto_adjust=True)

    results = {}
    for field in DATA_FIELDS:
        if field in raw.columns.get_level_values(0):
            df = raw[field].copy()
        else:
            logger.warning(f"Field '{field}' not found in downloaded data, skipping")
            continue

        df.index.name = "Date"
        df = df.dropna(axis=1, how="all")

        csv_path = output_path / f"{field}.csv"
        df.to_csv(csv_path)
        results[field] = df
        logger.info(f"Saved {field}.csv with shape {df.shape}")

    logger.info(f"Data saved to {output_path}")
    return results


def download_fx_pairs(
    start: str = "2012-01-01",
    end: str = "2025-12-31",
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Download FX pair history for country-ETF FX attribution (USD per CCY or CCY per USD)."""
    output_path = Path(output_dir) if output_dir else DATA_ROOT / "fx"
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for symbol, meta in FX_PAIRS.items():
        candidates = [symbol, *meta.get("fallbacks", [])]
        saved = False
        for candidate in candidates:
            try:
                raw = yf.download(candidate, start=start, end=end, auto_adjust=False, progress=False)
                if raw is None or len(raw) == 0:
                    continue
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                series = close.dropna()
                if series.empty:
                    continue
                series.index.name = "Date"
                # Save under the canonical symbol name expected by signal_report.py
                csv_path = output_path / f"{symbol}.csv"
                series.to_frame(name="Close").to_csv(csv_path)
                results[symbol] = series.to_frame(name="Close")
                logger.info(f"Saved {csv_path.name} with shape {results[symbol].shape}")
                saved = True
                break
            except Exception as e:
                logger.warning(f"FX download failed for {candidate}: {e}")
        if not saved:
            logger.warning(f"No data for FX pair {symbol} (tried {', '.join(candidates)})")

    logger.info(f"FX data saved to {output_path}")
    return results


def download_sp500_data(
    start: str = "2012-01-01",
    end: str = "2025-12-31",
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Backwards-compatible wrapper: download the S&P 500 universe."""
    return download_universe_data("sp500", start=start, end=end, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download price-volume data for signal discovery universes."
    )
    parser.add_argument(
        "--universe",
        default="sp500",
        choices=sorted(UNIVERSES),
        help="Universe preset to download. Default: sp500",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated custom ticker list (e.g. 'AAPL,MSFT,NVDA'). "
        "Overrides --universe for ticker selection and writes to data/custom/ by default.",
    )
    parser.add_argument(
        "--start", default="2012-01-01", help="Start date (YYYY-MM-DD). Default: 2012-01-01"
    )
    parser.add_argument(
        "--end", default="2025-12-31", help="End date (YYYY-MM-DD). Default: 2025-12-31"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Default: data/<universe>/ (data/sp500 for the default)",
    )
    parser.add_argument(
        "--download-fx",
        action="store_true",
        help="Download FX pairs for country-ETF local-currency attribution instead of an OHLCV universe "
        "(default output: data/fx/). See signal_report.py --local-currency.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.download_fx:
        download_fx_pairs(start=args.start, end=args.end, output_dir=args.output)
        return
    tickers = args.tickers.split(",") if args.tickers else None
    download_universe_data(
        universe=args.universe,
        start=args.start,
        end=args.end,
        output_dir=args.output,
        tickers=tickers,
    )


if __name__ == "__main__":
    main()
