import os
import urllib.request
import numpy as np
import polars as pl
import torch
from torch.utils.data import ConcatDataset, Dataset
from indicators import add_all_indicators

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
    # Commodities / Precious Metals
    "XAUUSD", "XAGUSD"
]

# Timeframe Lookback Sequence Lengths (Total L = 1440)
LEN_1D = 180    # 6 months of Daily candles
LEN_4H = 540    # 3 months of 4-Hour candles (6 candles/day * 90 days)
LEN_1H = 720    # 30 days of 1-Hour candles (24 candles/day * 30 days)
TOTAL_L = LEN_1D + LEN_4H + LEN_1H  # 1440 tokens

FEATURE_COLS = [
    # Cyclical sin/cos temporal features (8)
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "hour_open_sin",
    "hour_open_cos",
    "hour_end_sin",
    "hour_end_cos",

    # Relative price & volume returns (5)
    "feat_open_pct",
    "feat_close_pct",   # Primary close percentage return
    "feat_high_pct",
    "feat_low_pct",
    "feat_volume_pct",

    # Timeframe identifier (1.0 = Daily, 2.0 = 4H, 3.0 = 1H)
    "timeframe_id",

    # Technical Indicators from indicators.py (12)
    "feat_rsi",
    "feat_macd",
    "feat_macd_signal",
    "feat_macd_hist",
    "feat_bb_pos",
    "feat_atr",
    "feat_sma_50",
    "feat_sma_200",
    "feat_ema_10",
    "feat_ema_20",
    "feat_vma",
    "feat_vwap",
]


def ensure_datetime_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardizes timestamp / date column into a proper Polars Datetime type.
    """
    ts_cols = [c for c in df.columns if c.lower() in ["datetime", "timestamp", "time", "date"]]
    if not ts_cols:
        raise ValueError("No datetime or timestamp column found in dataset")

    col = ts_cols[0]
    if df[col].dtype in [pl.Int64, pl.Int32, pl.UInt64]:
        first_val = df[col][0]
        if first_val > 1e14:
            df = df.with_columns(pl.from_epoch(pl.col(col), time_unit="us").alias("datetime"))
        elif first_val > 1e11:
            df = df.with_columns(pl.from_epoch(pl.col(col), time_unit="ms").alias("datetime"))
        else:
            df = df.with_columns(pl.from_epoch(pl.col(col), time_unit="s").alias("datetime"))
    elif df[col].dtype in [pl.Utf8, pl.String]:
        df = df.with_columns(pl.col(col).str.to_datetime().alias("datetime"))
    elif "datetime" not in df.columns:
        df = df.rename({col: "datetime"})

    return df.sort("datetime")


def resample_ohlcv(df_1h: pl.DataFrame, interval: str) -> pl.DataFrame:
    """
    Resamples 1-Hour OHLCV into higher timeframes (4H or 1D) using Polars dynamic aggregation.
    """
    return df_1h.group_by_dynamic("datetime", every=interval).agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ]).sort("datetime")


def compute_timeframe_features(df: pl.DataFrame, timeframe_id: float) -> pl.DataFrame:
    """
    Computes all technical indicators (indicators.py), cyclical sin/cos temporal features,
    and stationary relative returns for any timeframe.
    """
    # 1. Compute and normalize all technical indicators from indicators.py
    df = add_all_indicators(df)

    prev_close = pl.col("close").shift(1)
    prev_vol = pl.col("volume").shift(1)
    pi2 = 2.0 * np.pi

    df = df.with_columns([
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.day().alias("day"),
        pl.col("datetime").dt.hour().alias("hour_open"),
        ((pl.col("datetime").dt.hour() + 1) % 24).alias("hour_end"),
    ])

    df_featured = df.with_columns([
        # Cyclical features
        (pi2 * pl.col("month") / 12.0).sin().alias("month_sin"),
        (pi2 * pl.col("month") / 12.0).cos().alias("month_cos"),
        (pi2 * pl.col("day") / 31.0).sin().alias("day_sin"),
        (pi2 * pl.col("day") / 31.0).cos().alias("day_cos"),
        (pi2 * pl.col("hour_open") / 24.0).sin().alias("hour_open_sin"),
        (pi2 * pl.col("hour_open") / 24.0).cos().alias("hour_open_cos"),
        (pi2 * pl.col("hour_end") / 24.0).sin().alias("hour_end_sin"),
        (pi2 * pl.col("hour_end") / 24.0).cos().alias("hour_end_cos"),

        # Stationary relative price & volume returns
        ((pl.col("open") - prev_close) / prev_close).alias("feat_open_pct"),
        ((pl.col("close") - prev_close) / prev_close).alias("feat_close_pct"),
        ((pl.col("high") - prev_close) / prev_close).alias("feat_high_pct"),
        ((pl.col("low") - prev_close) / prev_close).alias("feat_low_pct"),
        (((pl.col("volume") - prev_vol) / (prev_vol + 1e-8)).clip(-1.0, 1.0)).alias("feat_volume_pct"),

        # Timeframe Tag (1.0 = 1D, 2.0 = 4H, 3.0 = 1H)
        pl.lit(timeframe_id).alias("timeframe_id"),
        pl.col("close").alias("nominal_close")
    ]).drop_nulls()

    return df_featured


def load_pair_data(pair: str, data_dir: str = "data") -> pl.DataFrame:
    """
    Loads 1-Hour parquet data from local cache or downloads from Hugging Face if missing.
    """
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

    col_map = {c: c.lower() for c in df.columns if c.lower() in ["open", "high", "low", "close", "volume", "time", "timestamp", "datetime"]}
    df = df.rename(col_map)
    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("volume"))

    return ensure_datetime_column(df)


class MultiTimeframeDataset(Dataset):
    """
    Hierarchical Multi-Timeframe Dataset producing context windows of length L = 1440:
        - 180 Daily (1D) candles
        - 540 4-Hour (4H) candles
        - 720 1-Hour (1H) candles
    
    Target:
        - Next 1-Hour candle return vector of shape (feats=14,)
    """
    def __init__(
        self,
        features_1d: np.ndarray,
        features_4h: np.ndarray,
        features_1h: np.ndarray,
        nominal_closes_1h: np.ndarray,
        h1_indices: np.ndarray,
        idx_1d_map: np.ndarray,
        idx_4h_map: np.ndarray,
        pair_name: str,
    ):
        super().__init__()
        self.features_1d = features_1d
        self.features_4h = features_4h
        self.features_1h = features_1h
        self.nominal_closes_1h = nominal_closes_1h
        self.h1_indices = h1_indices
        self.idx_1d_map = idx_1d_map
        self.idx_4h_map = idx_4h_map
        self.pair_name = pair_name

    def __len__(self) -> int:
        return len(self.h1_indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        h1_idx = self.h1_indices[i]
        d1_idx = self.idx_1d_map[i]
        h4_idx = self.idx_4h_map[i]

        # 1. Slice past 180 Daily candles: shape (180, 14)
        c_1d = self.features_1d[d1_idx - LEN_1D + 1 : d1_idx + 1]

        # 2. Slice past 540 4-Hour candles: shape (540, 14)
        c_4h = self.features_4h[h4_idx - LEN_4H + 1 : h4_idx + 1]

        # 3. Slice past 720 1-Hour candles: shape (720, 14)
        c_1h = self.features_1h[h1_idx - LEN_1H + 1 : h1_idx + 1]

        # 4. Concatenate along time: Total sequence length L = 180 + 540 + 720 = 1440
        x = np.concatenate([c_1d, c_4h, c_1h], axis=0)

        # 5. Target: Next 1-Hour candle return vector
        target_next = self.features_1h[h1_idx + 1]
        prev_close = self.nominal_closes_1h[h1_idx]
        actual_next_close = self.nominal_closes_1h[h1_idx + 1]

        return {
            "x": torch.from_numpy(x),                                     # (1440, 14)
            "target_next": torch.from_numpy(target_next),                 # (14,)
            "prev_close": torch.tensor(prev_close, dtype=torch.float32),  # scalar
            "actual_next_close": torch.tensor(actual_next_close, dtype=torch.float32),
            "pair": self.pair_name
        }


def process_pair_multi_timeframe(
    df_raw: pl.DataFrame,
    pair_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    step_stride: int = 4,
) -> tuple[MultiTimeframeDataset, MultiTimeframeDataset, MultiTimeframeDataset] | None:
    """
    Resamples 1H data into 4H and 1D, computes features, performs strictly causal
    timeframe alignment via join_asof, and builds sequential train/val/test datasets.
    """
    # 1. Resample to 4H and 1D
    df_4h_raw = resample_ohlcv(df_raw, "4h")
    df_1d_raw = resample_ohlcv(df_raw, "1d")

    # 2. Compute 14 features for each timeframe
    df_1h = compute_timeframe_features(df_raw, timeframe_id=3.0)
    df_4h = compute_timeframe_features(df_4h_raw, timeframe_id=2.0)
    df_1d = compute_timeframe_features(df_1d_raw, timeframe_id=1.0)

    # 3. Add closed_at and row index for causal alignment
    # Match the exact datetime dtype to prevent Polars SchemaError in join_asof
    dt_type = df_1h["datetime"].dtype
    df_1d = df_1d.with_columns([
        (pl.col("datetime") + pl.duration(days=1)).cast(dt_type).alias("closed_at"),
        pl.int_range(0, pl.len()).alias("idx_1d")
    ])
    df_4h = df_4h.with_columns([
        (pl.col("datetime") + pl.duration(hours=4)).cast(dt_type).alias("closed_at"),
        pl.int_range(0, pl.len()).alias("idx_4h")
    ])

    # 4. Causal join_asof: match 1H candle with the latest CLOSED 1D and 4H candle
    df_aligned = df_1h.join_asof(
        df_1d.select(["closed_at", "idx_1d"]),
        left_on="datetime",
        right_on="closed_at",
        strategy="backward"
    ).join_asof(
        df_4h.select(["closed_at", "idx_4h"]),
        left_on="datetime",
        right_on="closed_at",
        strategy="backward"
    ).with_columns(
        pl.int_range(0, pl.len()).alias("h1_idx")
    )

    # 5. Filter valid indices with full lookback:
    # Need 180 1D candles, 540 4H candles, 720 1H candles, and 1 forward candle for target
    valid_df = df_aligned.filter(
        (pl.col("idx_1d") >= LEN_1D - 1) &
        (pl.col("idx_4h") >= LEN_4H - 1) &
        (pl.col("h1_idx") >= LEN_1H - 1) &
        (pl.col("h1_idx") < len(df_1h) - 1)
    )

    if len(valid_df) < 100:
        return None

    # Pre-extract arrays for zero-copy numpy slicing in DataLoader
    feats_1d = df_1d.select(FEATURE_COLS).to_numpy().astype(np.float32)
    feats_4h = df_4h.select(FEATURE_COLS).to_numpy().astype(np.float32)
    feats_1h = df_1h.select(FEATURE_COLS).to_numpy().astype(np.float32)
    nominal_closes = df_1h.select("nominal_close").to_numpy().astype(np.float32).flatten()

    # Apply step_stride (e.g. sample every 4 hours instead of every 1 hour to reduce 99.9% redundant overlap)
    h1_indices = valid_df.select("h1_idx").to_numpy().flatten()[::step_stride]
    idx_1d_map = valid_df.select("idx_1d").to_numpy().flatten()[::step_stride]
    idx_4h_map = valid_df.select("idx_4h").to_numpy().flatten()[::step_stride]

    # 6. Sequential train / val / test split
    total = len(h1_indices)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    def make_sub_dataset(start: int, end: int):
        return MultiTimeframeDataset(
            features_1d=feats_1d,
            features_4h=feats_4h,
            features_1h=feats_1h,
            nominal_closes_1h=nominal_closes,
            h1_indices=h1_indices[start:end],
            idx_1d_map=idx_1d_map[start:end],
            idx_4h_map=idx_4h_map[start:end],
            pair_name=pair_name,
        )

    train_ds = make_sub_dataset(0, train_end)
    val_ds = make_sub_dataset(train_end, val_end)
    test_ds = make_sub_dataset(val_end, total)

    return train_ds, val_ds, test_ds


def build_multi_pair_datasets(
    pairs: list[str] | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    step_stride: int = 4,
    data_dir: str = "data"
) -> tuple[ConcatDataset, ConcatDataset, ConcatDataset, list[str]]:
    """
    Builds unified multi-pair datasets with context length L = 1440 across all specified pairs.
    """
    if pairs is None:
        pairs = ALL_PAIRS

    train_sets = []
    val_sets = []
    test_sets = []
    loaded_pairs = []

    print(f"Building Multi-Timeframe Datasets (L = {TOTAL_L} tokens: 180 1D + 540 4H + 720 1H | Stride: {step_stride}h) across {len(pairs)} pairs...")
    for pair in pairs:
        try:
            df_raw = load_pair_data(pair, data_dir=data_dir)
            res = process_pair_multi_timeframe(
                df_raw=df_raw,
                pair_name=pair,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                step_stride=step_stride,
            )
            if res is None:
                continue

            t_ds, v_ds, te_ds = res
            train_sets.append(t_ds)
            val_sets.append(v_ds)
            test_sets.append(te_ds)
            loaded_pairs.append(pair)
        except Exception as e:
            print(f"Skipping {pair}: {e}")

    if not train_sets:
        raise RuntimeError("No valid multi-timeframe datasets could be built.")

    return ConcatDataset(train_sets), ConcatDataset(val_sets), ConcatDataset(test_sets), loaded_pairs
