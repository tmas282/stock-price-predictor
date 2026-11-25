import os
import urllib.request
from datetime import datetime, timezone
import numpy as np
import polars as pl
import torch
from torch.utils.data import ConcatDataset, Dataset
from candles.indicators import add_all_indicators

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

# Daily (1D) lookback & forecast horizons
LEN_1D = 180    # 180 Daily candles (~6 months)
PRED_DAYS = 7   # Next 7 Daily candles

FEATURE_COLS = [
    # Cyclical sin/cos daily temporal features (6) — bounded in [-1, 1]
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "dow_sin",
    "dow_cos",

    # Raw OHLCV (5) — Min-Max scaled per 180-day window
    "open",
    "high",
    "low",
    "close",
    "volume",

    # Price-level technical indicators (6) — scaled with shared window (p_min, p_range)
    "SMA_50",
    "SMA_200",
    "EMA_10",
    "EMA_20",
    "VMA_20",
    "VWAP",

    # Non-price technical indicators (6) — Min-Max scaled per column across 180-day window
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Pos",
    "ATR_14",
]

OHLC_COL_IDXS = [FEATURE_COLS.index(c) for c in ("open", "high", "low", "close")]
VOL_COL_IDX = FEATURE_COLS.index("volume")
PRICE_LEVEL_IND_IDXS = [FEATURE_COLS.index(c) for c in ("SMA_50", "SMA_200", "EMA_10", "EMA_20", "VMA_20", "VWAP")]
MINMAX_IND_IDXS = [FEATURE_COLS.index(c) for c in ("RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "BB_Pos", "ATR_14")]

TARGET_COLS = ["open", "high", "low", "close", "volume"]
TARGET_DIM = len(TARGET_COLS)

TARGET_OPEN_IDX = 0
TARGET_HIGH_IDX = 1
TARGET_LOW_IDX = 2
TARGET_CLOSE_IDX = 3
TARGET_VOLUME_IDX = 4


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


def resample_to_daily(df_1h: pl.DataFrame) -> pl.DataFrame:
    """
    Resamples 1-Hour OHLCV into Daily (1D) candles using Polars dynamic aggregation.
    """
    return df_1h.group_by_dynamic("datetime", every="1d").agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ]).sort("datetime")


def compute_daily_features(df_1d: pl.DataFrame) -> pl.DataFrame:
    """
    Computes raw technical indicators and cyclical daily temporal features.
    """
    df = add_all_indicators(df_1d)
    pi2 = 2.0 * np.pi

    df_featured = df.with_columns([
        (pi2 * pl.col("datetime").dt.month() / 12.0).sin().alias("month_sin"),
        (pi2 * pl.col("datetime").dt.month() / 12.0).cos().alias("month_cos"),
        (pi2 * pl.col("datetime").dt.day() / 31.0).sin().alias("day_sin"),
        (pi2 * pl.col("datetime").dt.day() / 31.0).cos().alias("day_cos"),
        (pi2 * (pl.col("datetime").dt.weekday() - 1) / 7.0).sin().alias("dow_sin"),
        (pi2 * (pl.col("datetime").dt.weekday() - 1) / 7.0).cos().alias("dow_cos"),
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


class DailyDataset(Dataset):
    """
    Daily (1D) sliding-window Dataset for iTransformer:
        - Input x: (LEN_1D=180, feats=23)
        - Target:  (PRED_DAYS=7, 5) next 7 Daily [open, high, low, close, volume]

    Normalization:
        - Computes causal window Min-Max across all 180 Daily input candles:
          * OHLC (and price-level indicators) share (p_min, p_range)
          * Volume uses (v_min, v_range)
          * Non-price indicators are Min-Max scaled per column into [0, 1]
        - Target is normalized with the exact same (p_min, p_range) and (v_min, v_range).
    """
    def __init__(
        self,
        features_1d: np.ndarray,
        ohlcv_1d: np.ndarray,
        d1_indices: np.ndarray,
        pair_name: str,
    ):
        super().__init__()
        self.features_1d = features_1d
        self.ohlcv_1d = ohlcv_1d
        self.d1_indices = d1_indices
        self.pair_name = pair_name

    def __len__(self) -> int:
        return len(self.d1_indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        d1_idx = self.d1_indices[i]

        # 1. Slice past 180 Daily candles: shape (180, feats)
        x = self.features_1d[d1_idx - LEN_1D + 1 : d1_idx + 1].copy()

        # 2. Window Min-Max scaling across all 180 input candles
        ohlc_vals = x[:, OHLC_COL_IDXS]
        p_min = float(ohlc_vals.min())
        p_max = float(ohlc_vals.max())
        p_range = float(max(p_max - p_min, 1e-8))

        vol_vals = x[:, VOL_COL_IDX]
        v_min = float(vol_vals.min())
        v_max = float(vol_vals.max())
        v_range = float(max(v_max - v_min, 1e-8))

        x[:, OHLC_COL_IDXS] = (ohlc_vals - p_min) / p_range
        x[:, PRICE_LEVEL_IND_IDXS] = np.clip((x[:, PRICE_LEVEL_IND_IDXS] - p_min) / p_range, -5.0, 5.0)
        x[:, VOL_COL_IDX] = (vol_vals - v_min) / v_range

        ind_vals = x[:, MINMAX_IND_IDXS]
        ind_min = ind_vals.min(axis=0)
        ind_max = ind_vals.max(axis=0)
        ind_range = np.maximum(ind_max - ind_min, 1e-8)
        x[:, MINMAX_IND_IDXS] = (ind_vals - ind_min) / ind_range

        # 3. Target: Next 7 Daily candles normalized with the SAME input Min-Max
        next_ohlcv = self.ohlcv_1d[d1_idx + 1 : d1_idx + 1 + PRED_DAYS].copy()
        target_next = np.empty((PRED_DAYS, TARGET_DIM), dtype=np.float32)
        target_next[:, 0:4] = (next_ohlcv[:, 0:4] - p_min) / p_range
        target_next[:, 4] = (next_ohlcv[:, 4] - v_min) / v_range
        target_next = np.clip(target_next, -5.0, 5.0)

        prev_close = self.ohlcv_1d[d1_idx, 3]

        return {
            "x": torch.from_numpy(x),                                     # (180, feats)
            "target_next": torch.from_numpy(target_next),                 # (7, 5)
            "actual_next_ohlcv": torch.from_numpy(next_ohlcv),            # (7, 5)
            "prev_close": torch.tensor(prev_close, dtype=torch.float32),  # scalar
            "p_min": torch.tensor(p_min, dtype=torch.float32),            # scalar
            "p_range": torch.tensor(p_range, dtype=torch.float32),        # scalar
            "v_min": torch.tensor(v_min, dtype=torch.float32),            # scalar
            "v_range": torch.tensor(v_range, dtype=torch.float32),        # scalar
            "pair": self.pair_name,
        }


def process_pair_daily(
    df_raw: pl.DataFrame,
    pair_name: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    step_stride: int = 1,
    min_date: str | None = None,
) -> tuple[DailyDataset, DailyDataset, DailyDataset] | None:
    """
    Resamples 1H data into Daily (1D), computes features, and builds
    sequential train/val/test DailyDatasets.
    """
    df_1d = compute_daily_features(resample_to_daily(df_raw)).with_columns(
        pl.int_range(0, pl.len()).alias("d1_idx")
    )

    valid_df = df_1d.filter(
        (pl.col("d1_idx") >= LEN_1D - 1) &
        (pl.col("d1_idx") < len(df_1d) - PRED_DAYS)
    )

    if min_date:
        dt_type = df_1d["datetime"].dtype
        min_dt = datetime.fromisoformat(min_date)
        dt_tz = getattr(dt_type, "time_zone", None)
        if dt_tz is not None and min_dt.tzinfo is None:
            min_dt = min_dt.replace(tzinfo=timezone.utc)
        elif dt_tz is None and min_dt.tzinfo is not None:
            min_dt = min_dt.astimezone(timezone.utc).replace(tzinfo=None)
        valid_df = valid_df.filter(pl.col("datetime") >= pl.lit(min_dt).cast(dt_type))

    if len(valid_df) < 30:
        return None

    feats_1d = df_1d.select(FEATURE_COLS).to_numpy().astype(np.float32)
    ohlcv_1d = df_1d.select(TARGET_COLS).to_numpy().astype(np.float32)
    d1_indices = valid_df.select("d1_idx").to_numpy().flatten()[::step_stride]

    total = len(d1_indices)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_ds = DailyDataset(feats_1d, ohlcv_1d, d1_indices[:train_end], pair_name)
    val_ds = DailyDataset(feats_1d, ohlcv_1d, d1_indices[train_end:val_end], pair_name)
    test_ds = DailyDataset(feats_1d, ohlcv_1d, d1_indices[val_end:], pair_name)

    return train_ds, val_ds, test_ds


def build_multi_pair_datasets(
    pairs: list[str] | None = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    step_stride: int = 1,
    data_dir: str = "data",
    min_date: str | None = None,
) -> tuple[ConcatDataset, ConcatDataset, ConcatDataset, list[str]]:
    """
    Builds unified multi-pair Daily (1D) datasets across all specified pairs.
    """
    if pairs is None:
        pairs = ALL_PAIRS

    train_sets = []
    val_sets = []
    test_sets = []
    loaded_pairs = []

    print(f"Building Daily (1D) Datasets (T = {LEN_1D} days -> S = {PRED_DAYS} days | Stride: {step_stride}d) across {len(pairs)} pairs...")
    for pair in pairs:
        try:
            df_raw = load_pair_data(pair, data_dir=data_dir)
            res = process_pair_daily(
                df_raw=df_raw,
                pair_name=pair,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                step_stride=step_stride,
                min_date=min_date,
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
        raise RuntimeError("No valid daily datasets could be built.")

    return ConcatDataset(train_sets), ConcatDataset(val_sets), ConcatDataset(test_sets), loaded_pairs
