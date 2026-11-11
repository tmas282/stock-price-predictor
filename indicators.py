import polars as pl


def add_sma(df: pl.DataFrame, windows=[50, 200]) -> pl.DataFrame:
    close = pl.col("close")
    return df.with_columns([close.rolling_mean(window_size=w).alias(f"SMA_{w}") for w in windows])


def add_ema(df: pl.DataFrame, spans=[10, 20]) -> pl.DataFrame:
    close = pl.col("close")
    return df.with_columns([close.ewm_mean(span=s).alias(f"EMA_{s}") for s in spans])


def add_vma(df: pl.DataFrame, window=20) -> pl.DataFrame:
    close = pl.col("close")
    vol = pl.col("volume")
    vol_sum = vol.rolling_sum(window_size=window).clip(lower_bound=1e-8)
    return df.with_columns([((close * vol).rolling_sum(window_size=window) / vol_sum).alias(f"VMA_{window}")])


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


def add_macd(df: pl.DataFrame, fast=12, slow=26, signal=9) -> pl.DataFrame:
    macd = pl.col("close").ewm_mean(span=fast) - pl.col("close").ewm_mean(span=slow)
    return df.with_columns([
        macd.alias("MACD"),
        macd.ewm_mean(span=signal).alias("MACD_Signal")
    ])


def add_bollinger_bands(df: pl.DataFrame, window=20, num_std=2) -> pl.DataFrame:
    middle = pl.col("close").rolling_mean(window_size=window)
    std = pl.col("close").rolling_std(window_size=window).clip(lower_bound=1e-8)
    return df.with_columns([
        middle.alias("BB_Middle"),
        (middle + num_std * std).alias("BB_Upper"),
        (middle - num_std * std).alias("BB_Lower")
    ])


def add_vwap(df: pl.DataFrame) -> pl.DataFrame:
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    vol_cum = pl.col("volume").cum_sum().clip(lower_bound=1e-8)
    return df.with_columns([((tp * pl.col("volume")).cum_sum() / vol_cum).alias("VWAP")])


def add_all_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes all technical indicators and scales them into stationary,
    scale-invariant features suitable for neural network input across all asset pairs.
    """
    # 1. Standardize column names to lowercase
    col_map = {c: c.lower() for c in df.columns if c.lower() in ["open", "high", "low", "close", "volume", "datetime", "timestamp", "time"]}
    df = df.rename(col_map)

    # 2. Compute raw indicators
    df = (
        df.pipe(add_sma, windows=[50, 200])
          .pipe(add_ema, spans=[10, 20])
          .pipe(add_vma, window=20)
          .pipe(add_atr, window=14)
          .pipe(add_rsi, window=14)
          .pipe(add_macd, fast=12, slow=26, signal=9)
          .pipe(add_bollinger_bands, window=20, num_std=2)
          .pipe(add_vwap)
    )

    close = pl.col("close")
    bb_range = (pl.col("BB_Upper") - pl.col("BB_Lower")).clip(lower_bound=1e-8)

    # 3. Create scale-invariant normalized feature columns
    df_normalized = df.with_columns([
        # RSI centered at 0: [-1.0, +1.0]
        ((pl.col("RSI_14") - 50.0) / 50.0).clip(-1.0, 1.0).alias("feat_rsi"),

        # MACD relative to price
        ((pl.col("MACD") / close) * 100.0).clip(-1.0, 1.0).alias("feat_macd"),
        ((pl.col("MACD_Signal") / close) * 100.0).clip(-1.0, 1.0).alias("feat_macd_signal"),
        (((pl.col("MACD") - pl.col("MACD_Signal")) / close) * 100.0).clip(-1.0, 1.0).alias("feat_macd_hist"),

        # Bollinger Bands %B position: [-1.0, +1.0]
        (((close - pl.col("BB_Middle")) / bb_range) * 2.0).clip(-2.0, 2.0).alias("feat_bb_pos"),

        # Relative volatility: ATR / price
        ((pl.col("ATR_14") / close) * 50.0).clip(0.0, 2.0).alias("feat_atr"),

        # Moving average relative distances: (Close - MA) / MA
        (((close - pl.col("SMA_50")) / pl.col("SMA_50")) * 20.0).clip(-1.0, 1.0).alias("feat_sma_50"),
        (((close - pl.col("SMA_200")) / pl.col("SMA_200")) * 20.0).clip(-1.0, 1.0).alias("feat_sma_200"),
        (((close - pl.col("EMA_10")) / pl.col("EMA_10")) * 50.0).clip(-1.0, 1.0).alias("feat_ema_10"),
        (((close - pl.col("EMA_20")) / pl.col("EMA_20")) * 50.0).clip(-1.0, 1.0).alias("feat_ema_20"),
        (((close - pl.col("VMA_20")) / pl.col("VMA_20")) * 20.0).clip(-1.0, 1.0).alias("feat_vma"),
        (((close - pl.col("VWAP")) / pl.col("VWAP")) * 20.0).clip(-1.0, 1.0).alias("feat_vwap"),
    ]).fill_null(strategy="backward").drop_nulls()

    return df_normalized
