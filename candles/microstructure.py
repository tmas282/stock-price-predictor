from __future__ import annotations

import numpy as np
import polars as pl

MICROSTRUCTURE_FEATURE_COLS = [
    "h1_realized_vol",
    "h1_parkinson_vol",
    "h1_rv_asym",
    "h1_efficiency",
    "h1_ret_asia",
    "h1_ret_london",
    "h1_ret_ny",
    "h1_body_ratio",
    "h1_close_pos",
    "h1_vol_imbalance",
]
MICROSTRUCTURE_MINMAX_COLS = ("h1_realized_vol", "h1_parkinson_vol")
MICROSTRUCTURE_ABSMAX_COLS = ("h1_ret_asia", "h1_ret_london", "h1_ret_ny")


def compute_daily_microstructure(df_1h: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates 1-Hour OHLCV bars into daily intraday microstructure features.
    Returns ['datetime', *MICROSTRUCTURE_FEATURE_COLS] sorted chronologically.
    """
    o = pl.col("open").cast(pl.Float64).clip(lower_bound=1e-8)
    h = pl.col("high").cast(pl.Float64).clip(lower_bound=1e-8)
    l = pl.col("low").cast(pl.Float64).clip(lower_bound=1e-8)
    c = pl.col("close").cast(pl.Float64).clip(lower_bound=1e-8)
    r = (c / o).log()
    r_sq = r.pow(2)
    r_abs = r.abs()
    hl_log_sq = (h / l).clip(lower_bound=1.0).log().pow(2)
    hour = pl.col("datetime").dt.hour()
    vol = pl.col("volume").cast(pl.Float64)
    signed_vol = (
        pl.when(pl.col("close") > pl.col("open"))
        .then(vol)
        .when(pl.col("close") < pl.col("open"))
        .then(-vol)
        .otherwise(0.0)
    )

    df_pre = df_1h.with_columns([
        r.alias("_r"),
        r_sq.alias("_r_sq"),
        r_abs.alias("_r_abs"),
        hl_log_sq.alias("_hl_log_sq"),
        vol.alias("_vol"),
        signed_vol.alias("_signed_vol"),
        hour.alias("_hour"),
    ])

    return (
        df_pre.group_by_dynamic("datetime", every="1d")
        .agg([
            pl.col("open").first().cast(pl.Float64).alias("_d_open"),
            pl.col("high").max().cast(pl.Float64).alias("_d_high"),
            pl.col("low").min().cast(pl.Float64).alias("_d_low"),
            pl.col("close").last().cast(pl.Float64).alias("_d_close"),
            pl.col("_r_sq").sum().alias("_r_sq_sum"),
            pl.when(pl.col("_r") > 0).then(pl.col("_r_sq")).otherwise(0.0).sum().alias("_rs_pos"),
            pl.when(pl.col("_r") < 0).then(pl.col("_r_sq")).otherwise(0.0).sum().alias("_rs_neg"),
            pl.col("_r_abs").sum().alias("_r_abs_sum"),
            pl.col("_hl_log_sq").sum().alias("_hl_log_sq_sum"),
            pl.col("_vol").sum().alias("_vol_sum"),
            pl.col("_signed_vol").sum().alias("_signed_vol_sum"),
            pl.when(pl.col("_hour") < 7)
            .then(pl.col("_r"))
            .otherwise(0.0)
            .sum()
            .alias("h1_ret_asia"),
            pl.when((pl.col("_hour") >= 7) & (pl.col("_hour") < 13))
            .then(pl.col("_r"))
            .otherwise(0.0)
            .sum()
            .alias("h1_ret_london"),
            pl.when((pl.col("_hour") >= 13) & (pl.col("_hour") < 21))
            .then(pl.col("_r"))
            .otherwise(0.0)
            .sum()
            .alias("h1_ret_ny"),
        ])
        .with_columns([
            pl.col("_r_sq_sum").clip(lower_bound=0.0).sqrt().alias("h1_realized_vol"),
            (pl.col("_hl_log_sq_sum") / (4.0 * np.log(2.0)))
            .clip(lower_bound=0.0)
            .sqrt()
            .alias("h1_parkinson_vol"),
            pl.when(pl.col("_r_sq_sum") > 1e-12)
            .then((pl.col("_rs_pos") - pl.col("_rs_neg")) / pl.col("_r_sq_sum"))
            .otherwise(0.0)
            .clip(-1.0, 1.0)
            .alias("h1_rv_asym"),
            pl.when(pl.col("_r_abs_sum") > 1e-10)
            .then(
                (
                    pl.col("_d_close").clip(lower_bound=1e-8)
                    / pl.col("_d_open").clip(lower_bound=1e-8)
                )
                .log()
                .abs()
                / pl.col("_r_abs_sum")
            )
            .otherwise(0.0)
            .clip(0.0, 1.0)
            .alias("h1_efficiency"),
            pl.when((pl.col("_d_high") - pl.col("_d_low")) > 1e-8)
            .then((pl.col("_d_close") - pl.col("_d_open")) / (pl.col("_d_high") - pl.col("_d_low")))
            .otherwise(0.0)
            .clip(-1.0, 1.0)
            .alias("h1_body_ratio"),
            pl.when((pl.col("_d_high") - pl.col("_d_low")) > 1e-8)
            .then((pl.col("_d_close") - pl.col("_d_low")) / (pl.col("_d_high") - pl.col("_d_low")))
            .otherwise(0.5)
            .clip(0.0, 1.0)
            .alias("h1_close_pos"),
            pl.when(pl.col("_vol_sum") > 1e-8)
            .then(pl.col("_signed_vol_sum") / pl.col("_vol_sum"))
            .otherwise(0.0)
            .clip(-1.0, 1.0)
            .alias("h1_vol_imbalance"),
        ])
        .select(["datetime", *MICROSTRUCTURE_FEATURE_COLS])
        .sort("datetime")
    )
