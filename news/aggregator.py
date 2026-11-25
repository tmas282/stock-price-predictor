from __future__ import annotations

from pathlib import Path

import polars as pl

from news.currency_mapper import CURRENCY_LIST

_RAW_NEWS_COLS = ("sentiment", "importance", "p_pos", "p_neg")

NEWS_FEATURE_COLS = [
    "base_news_sentiment",
    "base_news_importance",
    "base_news_p_pos",
    "base_news_p_neg",
    "quote_news_sentiment",
    "quote_news_importance",
    "quote_news_p_pos",
    "quote_news_p_neg",
]
NEWS_MINMAX_COLS = ("base_news_importance", "quote_news_importance")


def aggregate_news_to_1d_currencies(
    df_news: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """
    Causally aggregates scored news (`data/news_sentiment.parquet`) to Daily (1D)
    buckets for each currency in CURRENCY_LIST.

    Causal rule:
      A 1D candle labeled `D 00:00:00` covers [D 00:00, D+1 00:00) and closes at
      `D+1 00:00`. Every article published on calendar day `D` (UTC) is known by
      the close of candle `D`.

    Returns:
      dict[currency, pl.DataFrame] with columns:
        ['datetime', 'sentiment', 'importance', 'p_pos', 'p_neg']
    """
    empty_schema = {
        "datetime": pl.Datetime("us", "UTC"),
        "sentiment": pl.Float32,
        "importance": pl.Float32,
        "p_pos": pl.Float32,
        "p_neg": pl.Float32,
    }
    if df_news.is_empty():
        return {ccy: pl.DataFrame(schema=empty_schema) for ccy in CURRENCY_LIST}

    df_agg = (
        df_news.with_columns(
            pl.col("published_utc").dt.truncate("1d").alias("datetime"),
            (pl.col("p_pos") * pl.col("importance")).alias("_w_pos"),
            (pl.col("p_neg") * pl.col("importance")).alias("_w_neg"),
        )
        .group_by(["datetime", "currency"])
        .agg([
            pl.col("weighted_sentiment").sum().alias("_w_sent_sum"),
            pl.col("_w_pos").sum().alias("_w_pos_sum"),
            pl.col("_w_neg").sum().alias("_w_neg_sum"),
            pl.col("importance").sum().alias("_imp_sum"),
        ])
        .with_columns([
            pl.when(pl.col("_imp_sum") > 0)
            .then(pl.col("_w_sent_sum") / pl.col("_imp_sum"))
            .otherwise(0.0)
            .clip(-1.0, 1.0)
            .cast(pl.Float32)
            .alias("sentiment"),
            pl.col("_imp_sum").log1p().cast(pl.Float32).alias("importance"),
            pl.when(pl.col("_imp_sum") > 0)
            .then(pl.col("_w_pos_sum") / pl.col("_imp_sum"))
            .otherwise(0.0)
            .clip(0.0, 1.0)
            .cast(pl.Float32)
            .alias("p_pos"),
            pl.when(pl.col("_imp_sum") > 0)
            .then(pl.col("_w_neg_sum") / pl.col("_imp_sum"))
            .otherwise(0.0)
            .clip(0.0, 1.0)
            .cast(pl.Float32)
            .alias("p_neg"),
        ])
        .select(["datetime", "currency", "sentiment", "importance", "p_pos", "p_neg"])
        .sort("datetime")
    )

    result: dict[str, pl.DataFrame] = {}
    for ccy in CURRENCY_LIST:
        df_ccy = df_agg.filter(pl.col("currency") == ccy).drop("currency")
        result[ccy] = df_ccy if not df_ccy.is_empty() else pl.DataFrame(schema=empty_schema)

    return result


def load_daily_news_map(news_path: str | Path) -> dict[str, pl.DataFrame]:
    """Loads scored news Parquet and returns daily per-currency aggregated DataFrames."""
    path = Path(news_path)
    if not path.exists():
        raise FileNotFoundError(
            f"News sentiment parquet not found at '{path}'. "
            "Run news_pipeline.py or disable news with USE_NEWS=false."
        )
    return aggregate_news_to_1d_currencies(pl.read_parquet(path))


def _align_datetime_dtype(df: pl.DataFrame, target_dtype: pl.DataType) -> pl.DataFrame:
    """Aligns `df['datetime']` time unit and timezone to `target_dtype`."""
    src_tz = getattr(df["datetime"].dtype, "time_zone", None)
    tgt_tz = getattr(target_dtype, "time_zone", None)
    expr = pl.col("datetime")
    if src_tz is not None and tgt_tz is None:
        expr = expr.dt.replace_time_zone(None)
    elif src_tz is None and tgt_tz is not None:
        expr = expr.dt.replace_time_zone(tgt_tz)
    elif src_tz != tgt_tz:
        expr = expr.dt.convert_time_zone(tgt_tz)
    return df.with_columns(expr.cast(target_dtype).alias("datetime"))


def attach_pair_news(
    df_target_1d: pl.DataFrame,
    pair_name: str,
    daily_news_map: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """
    Left-joins daily BASE and QUOTE currency news features onto `df_target_1d`,
    filling days without news with 0.0 so no candle rows are dropped.
    """
    base, quote = pair_name[:3], pair_name[3:]
    dt_type = df_target_1d["datetime"].dtype
    df_out = df_target_1d

    for prefix, ccy in (("base_news", base), ("quote_news", quote)):
        df_ccy = daily_news_map.get(ccy)
        if df_ccy is None or df_ccy.is_empty():
            df_out = df_out.with_columns([
                pl.lit(0.0, dtype=pl.Float32).alias(f"{prefix}_{col}")
                for col in _RAW_NEWS_COLS
            ])
            continue

        rename_map = {col: f"{prefix}_{col}" for col in _RAW_NEWS_COLS}
        df_ccy_aligned = _align_datetime_dtype(
            df_ccy.select(["datetime", *_RAW_NEWS_COLS]),
            dt_type,
        ).rename(rename_map)
        df_out = df_out.join(df_ccy_aligned, on="datetime", how="left")

    return df_out.with_columns([
        pl.col(col).fill_null(0.0).cast(pl.Float32) for col in NEWS_FEATURE_COLS
    ])
