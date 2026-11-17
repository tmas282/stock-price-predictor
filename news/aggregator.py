from __future__ import annotations

import numpy as np
import polars as pl

from news.currency_mapper import CURRENCY_LIST

LEN_NEWS_1H = 120  # 5 days * 24 hours
NEWS_FEATS = 12


def compute_sigmoidal_recency(
    length: int = LEN_NEWS_1H,
    midpoint: float = 36.0,
    scale: float = 12.0,
) -> np.ndarray:
    """
    Returns a (length,) float32 curve in chronological order:
      - oldest (length-1 hours ago) -> ~0.0
      - newest (0 hours ago)        -> ~1.0
    """
    hours_ago = (length - 1) - np.arange(length, dtype=np.float32)
    return (1.0 / (1.0 + np.exp((hours_ago - midpoint) / scale))).astype(np.float32)


def compute_hourly_time_features(datetimes_1h: pl.Series) -> np.ndarray:
    """
    Computes (T, 6) cyclical temporal features for each 1H bucket:
      [hour_sin, hour_cos, dow_sin, dow_cos, week_sin, week_cos]
    """
    pi2 = 2.0 * np.pi
    df_t = pl.DataFrame({"datetime": datetimes_1h}).select([
        (pi2 * pl.col("datetime").dt.hour() / 24.0).sin().alias("hour_sin"),
        (pi2 * pl.col("datetime").dt.hour() / 24.0).cos().alias("hour_cos"),
        (pi2 * (pl.col("datetime").dt.weekday() - 1) / 7.0).sin().alias("dow_sin"),
        (pi2 * (pl.col("datetime").dt.weekday() - 1) / 7.0).cos().alias("dow_cos"),
        (pi2 * (pl.col("datetime").dt.week() - 1) / 52.0).sin().alias("week_sin"),
        (pi2 * (pl.col("datetime").dt.week() - 1) / 52.0).cos().alias("week_cos"),
    ])
    return df_t.to_numpy().astype(np.float32)


def aggregate_news_to_1h_currencies(
    df_news: pl.DataFrame,
    datetimes_1h: pl.Series,
) -> dict[str, np.ndarray]:
    """
    Causally aggregates scored news (`data/news_sentiment.parquet`) onto the
    1H candle timeline `datetimes_1h` for each currency in CURRENCY_LIST.

    Causal rule:
      A 1H candle labeled `t` covers (t, t + 1h] and closes at `t + 1h`.
      Any article published at `published_utc` is assigned to
      `floor_1h(published_utc - 1us)`, so it is only visible once that
      candle has closed.

    Returns:
      dict[currency, np.ndarray of shape (len(datetimes_1h), 2)]
      where columns are [weighted_avg_sentiment, log1p_importance_sum].
    """
    n_steps = len(datetimes_1h)
    result = {ccy: np.zeros((n_steps, 2), dtype=np.float32) for ccy in CURRENCY_LIST}

    if df_news.is_empty() or n_steps == 0:
        return result

    target_dtype = datetimes_1h.dtype
    target_tz = getattr(target_dtype, "time_zone", None)

    # Align published_utc timezone to match datetimes_1h
    pub = df_news["published_utc"]
    pub_tz = getattr(pub.dtype, "time_zone", None)
    if pub_tz is not None and target_tz is None:
        df_work = df_news.with_columns(pl.col("published_utc").dt.replace_time_zone(None))
    elif pub_tz is None and target_tz is not None:
        df_work = df_news.with_columns(pl.col("published_utc").dt.replace_time_zone(target_tz))
    else:
        df_work = df_news

    # Map each article to the 1H candle bucket whose close >= published_utc
    df_agg = (
        df_work.with_columns(
            (pl.col("published_utc") - pl.duration(microseconds=1))
            .dt.truncate("1h")
            .cast(target_dtype)
            .alias("datetime")
        )
        .group_by(["datetime", "currency"])
        .agg([
            pl.col("weighted_sentiment").sum().alias("_w_sent_sum"),
            pl.col("importance").sum().alias("_imp_sum"),
        ])
        .with_columns([
            pl.when(pl.col("_imp_sum") > 0)
            .then(pl.col("_w_sent_sum") / pl.col("_imp_sum"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("sentiment"),
            pl.col("_imp_sum").log1p().cast(pl.Float32).alias("importance"),
        ])
    )

    base_grid = pl.DataFrame({"datetime": datetimes_1h})

    for ccy in CURRENCY_LIST:
        df_ccy = df_agg.filter(pl.col("currency") == ccy).select(["datetime", "sentiment", "importance"])
        if df_ccy.is_empty():
            continue
        joined = base_grid.join(df_ccy, on="datetime", how="left").with_columns([
            pl.col("sentiment").fill_null(0.0).cast(pl.Float32),
            pl.col("importance").fill_null(0.0).cast(pl.Float32),
        ])
        result[ccy] = joined.select(["sentiment", "importance"]).to_numpy().astype(np.float32)

    return result
