import polars as pl


def add_sma(df: pl.DataFrame, windows=[50, 200]) -> pl.DataFrame:
    close = pl.col("close")
    return df.with_columns([close.rolling_mean(window_size=w).alias(f"SMA_{w}") for w in windows])


def add_atr(df: pl.DataFrame, window=14) -> pl.DataFrame:
    tr = pl.max_horizontal([
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs()
    ])
    return df.with_columns(tr.rolling_mean(window_size=window).alias(f"ATR_{window}"))


def add_rsi(df: pl.DataFrame, window=14) -> pl.DataFrame:
    diff = pl.col("close").diff()
    gain = pl.when(diff > 0).then(diff).otherwise(0.0).rolling_mean(window_size=window)
    loss = pl.when(diff < 0).then(diff.abs()).otherwise(0.0).rolling_mean(window_size=window)
    rs = gain / (loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return df.with_columns(rsi.alias(f"RSI_{window}"))


def add_vwap(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    if "VWAP" in df.columns:
        return df
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    vol_sum = pl.col("volume").rolling_sum(window_size=window).clip(lower_bound=1e-8)
    return df.with_columns([((tp * pl.col("volume")).rolling_sum(window_size=window) / vol_sum).alias("VWAP")])


def add_all_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes all raw technical indicators for downstream window Min-Max scaling.
    """
    # 1. Standardize column names to lowercase
    col_map = {c: c.lower() for c in df.columns if c.lower() in ["open", "high", "low", "close", "volume", "datetime", "timestamp", "time"]}
    df = df.rename(col_map)

    # 2. Compute non-collinear raw indicators
    return (
        df.pipe(add_sma, windows=[50, 200])
          .pipe(add_atr, window=14)
          .pipe(add_rsi, window=14)
          .pipe(add_vwap)
          .fill_null(strategy="backward")
          .drop_nulls()
    )
