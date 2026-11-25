from __future__ import annotations

import os
import urllib.request

import numpy as np
import polars as pl

from candles.indicators import add_all_indicators
from candles.microstructure import compute_daily_microstructure

CURRENCIES = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"]

ALL_PAIRS = [
    # Major Pairs
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    # Euro Crosses
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    # Pound Crosses
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    # Yen Crosses
    "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
    # Other Crosses
    "AUDCAD", "AUDCHF", "AUDNZD", "CADCHF", "NZDCAD", "NZDCHF",
]
ALL_PAIRS_SET = set(ALL_PAIRS)

NUM_CROSS_PER_SIDE = len(CURRENCIES) - 2  # 6 base crosses + 6 quote crosses = 12
BASE_CROSS_COLS = [f"base_cross_{i}" for i in range(NUM_CROSS_PER_SIDE)]
QUOTE_CROSS_COLS = [f"quote_cross_{i}" for i in range(NUM_CROSS_PER_SIDE)]
CROSS_COLS = BASE_CROSS_COLS + QUOTE_CROSS_COLS

CANDLE_FEATURE_COLS = [
    # Cyclical sin/cos daily temporal features (6) — bounded in [-1, 1]
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "dow_sin",
    "dow_cos",

    # Target pair raw OHLCV (5) — Min-Max scaled per 180-day window
    "open",
    "high",
    "low",
    "close",
    "volume",

    # Target pair price-level technical indicators (3) — shared window (p_min, p_range)
    "SMA_50",
    "SMA_200",
    "VWAP",

    # Target pair non-price technical indicators (2) — Min-Max scaled per column
    "RSI_14",
    "ATR_14",

    # Correlated cross-pair closes (12): 6 oriented BASE/* + 6 oriented QUOTE/*
    *BASE_CROSS_COLS,
    *QUOTE_CROSS_COLS,
]

OHLC_COLS = ("open", "high", "low", "close")
PRICE_LEVEL_IND_COLS = ("SMA_50", "SMA_200", "VWAP")
MINMAX_CANDLE_COLS = ("RSI_14", "ATR_14", *CROSS_COLS)

TARGET_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
TARGET_CLOSE_IDX = TARGET_OHLCV_COLS.index("close")


def get_oriented_cross_specs(pair_name: str) -> list[tuple[str, str, bool]]:
    """
    For BASE/QUOTE, returns 12 specs (col_name, source_pair, invert) oriented so
    UP always reflects BASE (or QUOTE) strengthening against the other 6 currencies.
    """
    base, quote = pair_name[:3], pair_name[3:]
    others = [c for c in CURRENCIES if c not in (base, quote)]

    specs: list[tuple[str, str, bool]] = []
    for prefix, anchor in (("base_cross", base), ("quote_cross", quote)):
        for i, ccy in enumerate(others):
            direct, inverse = f"{anchor}{ccy}", f"{ccy}{anchor}"
            if direct in ALL_PAIRS_SET:
                specs.append((f"{prefix}_{i}", direct, False))
            elif inverse in ALL_PAIRS_SET:
                specs.append((f"{prefix}_{i}", inverse, True))
            else:
                raise ValueError(f"No cross pair found for {anchor}/{ccy}")

    return specs


def ensure_datetime_column(df: pl.DataFrame) -> pl.DataFrame:
    """Standardizes the timestamp/date column into a sorted Polars Datetime."""
    ts_cols = [c for c in df.columns if c.lower() in ["datetime", "timestamp", "time", "date"]]
    if not ts_cols:
        raise ValueError("No datetime or timestamp column found in dataset")

    col = ts_cols[0]
    if df[col].dtype in [pl.Int64, pl.Int32, pl.UInt64]:
        first_val = df[col][0]
        unit = "us" if first_val > 1e14 else ("ms" if first_val > 1e11 else "s")
        df = df.with_columns(pl.from_epoch(pl.col(col), time_unit=unit).alias("datetime"))
    elif df[col].dtype in [pl.Utf8, pl.String]:
        df = df.with_columns(pl.col(col).str.to_datetime().alias("datetime"))
    elif "datetime" not in df.columns:
        df = df.rename({col: "datetime"})

    return df.sort("datetime")


def load_pair_data(pair: str, data_dir: str = "data") -> pl.DataFrame:
    """Loads 1-Hour parquet data from local cache or downloads from Hugging Face."""
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, f"{pair}_H1.parquet")

    if os.path.exists(local_path):
        df = pl.read_parquet(local_path)
    else:
        hf_url = f"https://huggingface.co/datasets/Yllvar/fx-historical-data/resolve/main/{pair}/{pair}_H1.parquet"
        print(f"Downloading {pair} (H1) from Hugging Face...")
        try:
            urllib.request.urlretrieve(hf_url, local_path)
            df = pl.read_parquet(local_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to download {pair} (H1) from {hf_url}: {e}") from e

    col_map = {
        c: c.lower()
        for c in df.columns
        if c.lower() in ["open", "high", "low", "close", "volume", "time", "timestamp", "datetime"]
    }
    df = df.rename(col_map)
    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("volume"))

    return ensure_datetime_column(df)


def resample_to_daily(df_1h: pl.DataFrame) -> pl.DataFrame:
    """Resamples 1-Hour OHLCV into Daily (1D) candles with intraday daily VWAP."""
    tp = (
        pl.col("high").cast(pl.Float64)
        + pl.col("low").cast(pl.Float64)
        + pl.col("close").cast(pl.Float64)
    ) / 3.0
    vol = pl.col("volume").cast(pl.Float64)

    return (
        df_1h.with_columns([
            (tp * vol).alias("_tp_vol"),
            vol.alias("_vol"),
        ])
        .group_by_dynamic("datetime", every="1d")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("_tp_vol").sum().alias("_tp_vol_sum"),
            pl.col("_vol").sum().alias("_vol_sum"),
        ])
        .with_columns([
            pl.when(pl.col("_vol_sum") > 1e-8)
            .then(pl.col("_tp_vol_sum") / pl.col("_vol_sum"))
            .otherwise((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
            .alias("VWAP"),
        ])
        .drop(["_tp_vol_sum", "_vol_sum"])
        .sort("datetime")
    )


def compute_daily_features(df_1d: pl.DataFrame) -> pl.DataFrame:
    """Computes technical indicators and cyclical daily temporal features."""
    df = add_all_indicators(df_1d)
    pi2 = 2.0 * np.pi

    return df.with_columns([
        (pi2 * pl.col("datetime").dt.month() / 12.0).sin().alias("month_sin"),
        (pi2 * pl.col("datetime").dt.month() / 12.0).cos().alias("month_cos"),
        (pi2 * pl.col("datetime").dt.day() / 31.0).sin().alias("day_sin"),
        (pi2 * pl.col("datetime").dt.day() / 31.0).cos().alias("day_cos"),
        (pi2 * (pl.col("datetime").dt.weekday() - 1) / 7.0).sin().alias("dow_sin"),
        (pi2 * (pl.col("datetime").dt.weekday() - 1) / 7.0).cos().alias("dow_cos"),
    ]).drop_nulls()


def attach_oriented_cross_closes(
    df_target_1d: pl.DataFrame,
    pair_name: str,
    daily_closes_map: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Causally aligns the 12 oriented cross-pair daily closes via backward join_asof."""
    specs = get_oriented_cross_specs(pair_name)
    dt_type = df_target_1d["datetime"].dtype
    df_out = df_target_1d

    for col_name, src_pair, invert in specs:
        if src_pair not in daily_closes_map:
            raise KeyError(f"Missing daily close series for cross pair '{src_pair}' required by '{pair_name}'")

        df_cross = daily_closes_map[src_pair]
        close_expr = (1.0 / pl.col("close").clip(lower_bound=1e-8)) if invert else pl.col("close")
        df_cross_aligned = df_cross.select([
            pl.col("datetime").cast(dt_type).alias("cross_dt"),
            close_expr.cast(pl.Float32).alias(col_name),
        ]).sort("cross_dt")

        df_out = df_out.join_asof(
            df_cross_aligned,
            left_on="datetime",
            right_on="cross_dt",
            strategy="backward",
        ).drop("cross_dt")

    return df_out.drop_nulls(subset=CROSS_COLS)


def load_daily_candle_maps(
    pairs: list[str],
    data_dir: str = "data",
    include_microstructure: bool = True,
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """
    Loads all target pairs and their required cross pairs once, returning
    (daily_ohlcv_map, daily_closes_map).
    """
    target_pairs = set(pairs)
    needed_pairs = set(pairs)
    for pair in pairs:
        for _, src_pair, _ in get_oriented_cross_specs(pair):
            needed_pairs.add(src_pair)

    daily_ohlcv_map: dict[str, pl.DataFrame] = {}
    daily_closes_map: dict[str, pl.DataFrame] = {}
    for pair in sorted(needed_pairs):
        try:
            df_1h = load_pair_data(pair, data_dir=data_dir)
            df_1d = resample_to_daily(df_1h)
            if include_microstructure and pair in target_pairs:
                df_1d = df_1d.join(compute_daily_microstructure(df_1h), on="datetime", how="left")
            daily_ohlcv_map[pair] = df_1d
            daily_closes_map[pair] = df_1d.select(["datetime", "close"])
        except Exception as e:
            print(f"Warning: could not load {pair}: {e}")

    return daily_ohlcv_map, daily_closes_map


def build_pair_candle_frame(
    pair_name: str,
    daily_ohlcv_map: dict[str, pl.DataFrame],
    daily_closes_map: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Computes daily candle features and oriented cross closes for `pair_name`."""
    df_1d = compute_daily_features(daily_ohlcv_map[pair_name])
    return attach_oriented_cross_closes(df_1d, pair_name, daily_closes_map)
