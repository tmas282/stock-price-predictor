from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import torch
from torch.utils.data import ConcatDataset, Dataset

from candles.daily import (
    ALL_PAIRS,
    CANDLE_FEATURE_COLS,
    MINMAX_CANDLE_COLS,
    OHLC_COLS,
    PRICE_LEVEL_IND_COLS,
    TARGET_CLOSE_IDX,
    TARGET_OHLCV_COLS,
    build_pair_candle_frame,
    load_daily_candle_maps,
)
from news.aggregator import NEWS_FEATURE_COLS, NEWS_MINMAX_COLS, attach_pair_news, load_daily_news_map

# Daily (1D) lookback & forecast horizons
LEN_1D = 180    # 180 Daily candles (~6 months)
PRED_DAYS = 7   # Next 7 Daily candles
TARGET_DIM = 1  # Binary cumulative direction logit per forecast day: 1 if close[t+k] > close[t] else 0


def get_feature_cols(include_news: bool = False) -> list[str]:
    """Returns ordered model input feature columns with optional daily news."""
    if include_news:
        return [*CANDLE_FEATURE_COLS, *NEWS_FEATURE_COLS]
    return list(CANDLE_FEATURE_COLS)


def _filter_min_date(df: pl.DataFrame, min_date: str) -> pl.DataFrame:
    dt_type = df["datetime"].dtype
    min_dt = datetime.fromisoformat(min_date)
    dt_tz = getattr(dt_type, "time_zone", None)
    if dt_tz is not None and min_dt.tzinfo is None:
        min_dt = min_dt.replace(tzinfo=timezone.utc)
    elif dt_tz is None and min_dt.tzinfo is not None:
        min_dt = min_dt.astimezone(timezone.utc).replace(tzinfo=None)
    return df.filter(pl.col("datetime") >= pl.lit(min_dt).cast(dt_type))


class DailyDataset(Dataset):
    """
    Cross-Pair Daily (1D) sliding-window Dataset for iTransformer:
      - Input x: (LEN_1D=180, feats=35 or 43 with news)
      - Target:  (PRED_DAYS=7, 1) binary cumulative direction (close[t+k] > close[t])
    """

    def __init__(
        self,
        features_1d: np.ndarray,
        ohlcv_1d: np.ndarray,
        d1_indices: np.ndarray,
        pair_name: str,
        feature_cols: list[str],
    ):
        super().__init__()
        self.features_1d = features_1d
        self.ohlcv_1d = ohlcv_1d
        self.d1_indices = d1_indices
        self.pair_name = pair_name

        self.ohlc_idxs = [feature_cols.index(c) for c in OHLC_COLS]
        self.vol_idx = feature_cols.index("volume")
        self.price_ind_idxs = [feature_cols.index(c) for c in PRICE_LEVEL_IND_COLS]
        minmax_names = set(MINMAX_CANDLE_COLS) | set(NEWS_MINMAX_COLS)
        self.minmax_idxs = [i for i, c in enumerate(feature_cols) if c in minmax_names]

    def __len__(self) -> int:
        return len(self.d1_indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        d1_idx = self.d1_indices[i]

        # 1. Slice past 180 Daily candles
        x = self.features_1d[d1_idx - LEN_1D + 1 : d1_idx + 1].copy()

        # 2. Causal window Min-Max scaling across the 180 input candles
        ohlc_vals = x[:, self.ohlc_idxs]
        p_min = float(ohlc_vals.min())
        p_range = float(max(float(ohlc_vals.max()) - p_min, 1e-8))

        vol_vals = x[:, self.vol_idx]
        v_min = float(vol_vals.min())
        v_range = float(max(float(vol_vals.max()) - v_min, 1e-8))

        x[:, self.ohlc_idxs] = (ohlc_vals - p_min) / p_range
        x[:, self.price_ind_idxs] = np.clip((x[:, self.price_ind_idxs] - p_min) / p_range, -5.0, 5.0)
        x[:, self.vol_idx] = (vol_vals - v_min) / v_range

        col_vals = x[:, self.minmax_idxs]
        col_min = col_vals.min(axis=0)
        col_range = np.maximum(col_vals.max(axis=0) - col_min, 1e-8)
        x[:, self.minmax_idxs] = (col_vals - col_min) / col_range

        # 3. Target: Cumulative binary direction for next 7 Daily candles
        next_ohlcv = self.ohlcv_1d[d1_idx + 1 : d1_idx + 1 + PRED_DAYS].copy()
        prev_close = float(self.ohlcv_1d[d1_idx, TARGET_CLOSE_IDX])
        target_dir = (next_ohlcv[:, TARGET_CLOSE_IDX] > prev_close).astype(np.float32)[:, None]

        return {
            "x": torch.from_numpy(x),
            "target_dir": torch.from_numpy(target_dir),
            "actual_next_ohlcv": torch.from_numpy(next_ohlcv),
            "prev_close": torch.tensor(prev_close, dtype=torch.float32),
            "pair": self.pair_name,
        }


def process_pair_daily(
    pair_name: str,
    daily_ohlcv_map: dict[str, pl.DataFrame],
    daily_closes_map: dict[str, pl.DataFrame],
    daily_news_map: dict[str, pl.DataFrame] | None = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    step_stride: int = 1,
    min_date: str | None = None,
) -> tuple[DailyDataset, DailyDataset, DailyDataset] | None:
    """
    Builds sequential train/val/test DailyDatasets for `pair_name` from preloaded
    daily candle maps and optional daily currency news maps.
    """
    df_1d = build_pair_candle_frame(pair_name, daily_ohlcv_map, daily_closes_map)
    include_news = daily_news_map is not None
    if include_news:
        df_1d = attach_pair_news(df_1d, pair_name, daily_news_map)

    feature_cols = get_feature_cols(include_news=include_news)
    df_1d = df_1d.with_columns(pl.int_range(0, pl.len()).alias("d1_idx"))

    valid_df = df_1d.filter(
        (pl.col("d1_idx") >= LEN_1D - 1)
        & (pl.col("d1_idx") < len(df_1d) - PRED_DAYS)
    )
    if min_date:
        valid_df = _filter_min_date(valid_df, min_date)

    if len(valid_df) < 30:
        return None

    feats_1d = df_1d.select(feature_cols).to_numpy().astype(np.float32)
    ohlcv_1d = df_1d.select(TARGET_OHLCV_COLS).to_numpy().astype(np.float32)
    d1_indices = valid_df.select("d1_idx").to_numpy().flatten()[::step_stride]

    total = len(d1_indices)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    return (
        DailyDataset(feats_1d, ohlcv_1d, d1_indices[:train_end], pair_name, feature_cols),
        DailyDataset(feats_1d, ohlcv_1d, d1_indices[train_end:val_end], pair_name, feature_cols),
        DailyDataset(feats_1d, ohlcv_1d, d1_indices[val_end:], pair_name, feature_cols),
    )


def build_multi_pair_datasets(
    pairs: list[str] | None = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    step_stride: int = 1,
    data_dir: str = "data",
    min_date: str | None = None,
    news_path: str | None = None,
) -> tuple[ConcatDataset, ConcatDataset, ConcatDataset, list[str], list[str]]:
    """
    Builds unified multi-pair Cross-Pair Daily (1D) datasets across `pairs`.
    Includes optional daily currency news features when `news_path` is provided.
    """
    if pairs is None:
        pairs = ALL_PAIRS

    daily_ohlcv_map, daily_closes_map = load_daily_candle_maps(pairs, data_dir=data_dir)
    daily_news_map = load_daily_news_map(news_path) if news_path is not None else None
    feature_cols = get_feature_cols(include_news=daily_news_map is not None)

    mode_desc = f"candles + news ({news_path})" if daily_news_map is not None else "candles only"
    print(
        f"Building Cross-Pair Daily (1D) Datasets "
        f"(T={LEN_1D}d -> S={PRED_DAYS}d | {len(feature_cols)} feats [{mode_desc}] | Stride={step_stride}d) "
        f"across {len(pairs)} pairs..."
    )

    train_sets, val_sets, test_sets, loaded_pairs = [], [], [], []
    for pair in pairs:
        if pair not in daily_ohlcv_map:
            continue
        try:
            res = process_pair_daily(
                pair_name=pair,
                daily_ohlcv_map=daily_ohlcv_map,
                daily_closes_map=daily_closes_map,
                daily_news_map=daily_news_map,
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
        raise RuntimeError("No valid cross-pair daily datasets could be built.")

    return ConcatDataset(train_sets), ConcatDataset(val_sets), ConcatDataset(test_sets), loaded_pairs, feature_cols
