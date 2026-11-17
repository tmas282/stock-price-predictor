from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from news.currency_mapper import map_news_to_currencies
from news.finbert import FinBERTScorer
from news.gdelt_fetcher import fetch_gdelt_last_n_days, load_gkg_export


def run_news_pipeline(
    days: int = 5,
    input_gkg_path: str | Path | None = None,
    output_path: str | Path = "data/news_sentiment.parquet",
    min_importance: float = 0.25,
    batch_size: int = 128,
) -> pl.DataFrame:
    """
    End-to-end news pipeline:
      1. Fetch live last-N-days GDELT DOC 2.0 news (or load historical GKG export).
      2. Map headlines to FX currencies & importance weights.
      3. Run batched FinBERT inference and cache to Parquet.
    """
    if input_gkg_path is not None:
        print(f"Loading historical GKG export from {input_gkg_path}...")
        df_raw = load_gkg_export(input_gkg_path)
    else:
        print(f"Fetching last {days} days of FX/macro news from GDELT DOC 2.0...")
        df_raw = fetch_gdelt_last_n_days(days=days)

    print(f"Fetched {len(df_raw):,} unique articles.")
    df_mapped = map_news_to_currencies(df_raw, min_importance=min_importance)
    print(f"Mapped to {len(df_mapped):,} (article, currency) rows.")

    if df_mapped.is_empty():
        print("No FX-relevant articles found.")
        return df_mapped

    print("Running batched FinBERT inference...")
    scorer = FinBERTScorer(batch_size=batch_size)
    df_scored = scorer.score_dataframe(df_mapped)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.write_parquet(out_path)
    print(f"Saved {len(df_scored):,} scored rows to {out_path}")
    return df_scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT -> Currency Mapper -> Batch FinBERT Pipeline")
    parser.add_argument("--days", type=int, default=5, help="Lookback days for live GDELT DOC 2.0 fetch")
    parser.add_argument("--input", type=str, default=None, help="Optional historical GKG CSV/Parquet path")
    parser.add_argument("--output", type=str, default="data/news_sentiment.parquet", help="Output Parquet cache path")
    parser.add_argument("--min-importance", type=float, default=0.25, help="Minimum importance threshold")
    parser.add_argument("--batch-size", type=int, default=128, help="FinBERT inference batch size")
    args = parser.parse_args()

    df_result = run_news_pipeline(
        days=args.days,
        input_gkg_path=args.input,
        output_path=args.output,
        min_importance=args.min_importance,
        batch_size=args.batch_size,
    )
    if not df_result.is_empty():
        print("\nSample scored rows:")
        print(df_result.select(["published_utc", "currency", "importance", "net_sentiment", "title"]).head(10))
