from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from news.currency_mapper import map_news_to_currencies
from news.finbert import FinBERTScorer
from news.gdelt_fetcher import (
    GKG_PAGE_TITLE_START_DATE,
    clamp_gkg_start_date,
    estimate_historical_bigquery_cost,
    fetch_gdelt_last_n_days,
    fetch_historical_bigquery,
    load_gkg_export,
)


def _date_chunks(start_str: str, end_str: str, chunk_months: int) -> list[tuple[str, str]]:
    """Splits [start_str, end_str) into chronological chunks of `chunk_months` months."""
    start_d = date.fromisoformat(start_str)
    end_d = date.fromisoformat(end_str)
    if start_d >= end_d:
        return []
    if chunk_months <= 0:
        return [(start_str, end_str)]

    chunks: list[tuple[str, str]] = []
    cur = start_d
    while cur < end_d:
        total_months = cur.year * 12 + (cur.month - 1) + chunk_months
        nxt_year, nxt_month = divmod(total_months, 12)
        nxt_month += 1
        # Clamp day to valid calendar day
        nxt = date(nxt_year, nxt_month, 1)
        if nxt > end_d:
            nxt = end_d
        chunks.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return chunks


def _merge_and_save(df_new: pl.DataFrame, out_path: Path, merge: bool) -> pl.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if merge and out_path.exists():
        df_existing = pl.read_parquet(out_path)
        df_combined = pl.concat([df_existing, df_new], how="vertical_relaxed")
    else:
        df_combined = df_new

    # Sort chronologically first so deduplication always keeps the earliest
    # publication timestamp (preventing syndicated reprints from leaking later spikes).
    df_out = (
        df_combined.sort(["published_utc", "currency"])
        .unique(subset=["url", "currency"], keep="first")
        .with_columns(
            pl.col("title").str.to_lowercase().str.strip_chars().alias("_title_norm")
        )
        .unique(subset=["_title_norm", "currency"], keep="first")
        .drop("_title_norm")
        .sort(["published_utc", "currency"])
    )
    df_out.write_parquet(out_path)
    return df_out


def run_news_pipeline(
    days: int = 5,
    chunk_hours: int = 24,
    input_gkg_path: str | Path | None = None,
    historical: bool = False,
    start_date: str = GKG_PAGE_TITLE_START_DATE,
    end_date: str | None = None,
    chunk_months: int = 12,
    since_last: bool = False,
    dry_run: bool = False,
    max_gb_billed: float | None = None,
    gcp_project: str | None = None,
    save_raw_gkg: str | Path | None = None,
    output_path: str | Path = "data/news_sentiment.parquet",
    merge: bool = False,
    min_importance: float = 0.25,
    batch_size: int = 128,
) -> pl.DataFrame:
    """
    End-to-end news pipeline:
      1. Fetch news:
         - `input_gkg_path`: load a local BigQuery GKG CSV/Parquet export
         - `historical=True`: query GDELT 2.0 GKG (2019-09-22 -> present) via BigQuery
           in resumable `chunk_months` chunks (or `--dry-run` to inspect cost at $0)
         - otherwise: fetch last `days` days from GDELT DOC 2.0 API (chunked by `chunk_hours`)
      2. Map headlines to FX currencies & importance weights.
      3. Run batched FinBERT inference and save (or merge) to Parquet.
    """
    load_dotenv()
    if gcp_project is None:
        gcp_project = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")

    out_path = Path(output_path)

    if historical:
        eff_end = end_date or (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        eff_start = clamp_gkg_start_date(start_date)
        if since_last and out_path.exists():
            max_ts = pl.read_parquet(out_path, columns=["published_utc"])["published_utc"].max()
            if isinstance(max_ts, (date, datetime)):
                eff_start = clamp_gkg_start_date(max_ts.strftime("%Y-%m-%d"))
                merge = True
                print(f"[since-last] Resuming BigQuery fetch from {eff_start} -> {eff_end}")

        if dry_run:
            est = estimate_historical_bigquery_cost(
                start_date=eff_start,
                end_date=eff_end,
                project_id=gcp_project,
            )
            print(
                f"[BigQuery Dry Run] {eff_start} -> {eff_end}: "
                f"{est['gib']:.2f} GiB ({est['tib']:.4f} TiB) | "
                f"Est. cost: ${est['est_usd']:.2f} (first 1 TiB/month is free)"
            )
            return pl.DataFrame()

        chunks = _date_chunks(eff_start, eff_end, chunk_months)
        if not chunks:
            print(f"Empty date range: {eff_start} -> {eff_end}")
            return pl.DataFrame()

        scorer: FinBERTScorer | None = None
        df_total = pl.DataFrame()
        raw_dir = Path(save_raw_gkg) if save_raw_gkg is not None else None
        remaining_gb: float | None = (
            float(max_gb_billed) if (max_gb_billed is not None and max_gb_billed > 0) else None
        )
        total_scanned_gib = 0.0

        for idx, (c_start, c_end) in enumerate(chunks, start=1):
            chunk_est = estimate_historical_bigquery_cost(
                start_date=c_start,
                end_date=c_end,
                project_id=gcp_project,
            )
            chunk_gib = float(chunk_est["gib"])
            if remaining_gb is not None and chunk_gib > remaining_gb:
                print(
                    f"\n[Budget Stop] Chunk {idx}/{len(chunks)} ({c_start} -> {c_end}) requires "
                    f"{chunk_gib:.2f} GiB, exceeding remaining budget ({remaining_gb:.2f} GiB of "
                    f"{max_gb_billed:.2f} GiB total; {total_scanned_gib:.2f} GiB used so far). "
                    f"Re-run with --since-last when ready to continue."
                )
                break

            print(
                f"\n[Chunk {idx}/{len(chunks)}] Fetching BigQuery GKG {c_start} -> {c_end} "
                f"(est. {chunk_gib:.2f} GiB)..."
            )
            raw_out = (
                raw_dir / f"gkg_{c_start}_{c_end}.parquet"
                if (raw_dir is not None and len(chunks) > 1)
                else raw_dir
            )
            df_raw = fetch_historical_bigquery(
                start_date=c_start,
                end_date=c_end,
                project_id=gcp_project,
                output_path=raw_out,
                max_gb_billed=remaining_gb,
            )
            total_scanned_gib += chunk_gib
            if remaining_gb is not None:
                remaining_gb = max(0.0, remaining_gb - chunk_gib)
            print(f"  Fetched {len(df_raw):,} unique English macro articles.")
            df_mapped = map_news_to_currencies(df_raw, min_importance=min_importance)
            print(f"  Mapped to {len(df_mapped):,} (article, currency) rows.")
            if df_mapped.is_empty():
                continue

            if scorer is None:
                print("  Loading FinBERT...")
                scorer = FinBERTScorer(batch_size=batch_size)
            df_scored = scorer.score_dataframe(df_mapped)
            should_merge = merge or (idx > 1)
            df_total = _merge_and_save(df_scored, out_path, merge=should_merge)
            print(f"  Saved checkpoint -> {len(df_total):,} total rows in {out_path}")

        return df_total

    if input_gkg_path is not None:
        print(f"Loading historical GKG export from {input_gkg_path}...")
        df_raw = load_gkg_export(input_gkg_path)
    else:
        print(f"Fetching last {days} days of FX/macro news from GDELT DOC 2.0 (chunk={chunk_hours}h)...")
        df_raw = fetch_gdelt_last_n_days(days=days, chunk_hours=chunk_hours)

    print(f"Fetched {len(df_raw):,} unique articles.")
    df_mapped = map_news_to_currencies(df_raw, min_importance=min_importance)
    print(f"Mapped to {len(df_mapped):,} (article, currency) rows.")

    if df_mapped.is_empty():
        print("No FX-relevant articles found.")
        return df_mapped

    print("Running batched FinBERT inference...")
    scorer = FinBERTScorer(batch_size=batch_size)
    df_scored = scorer.score_dataframe(df_mapped)

    df_scored = _merge_and_save(df_scored, out_path, merge=merge)
    print(f"Saved {len(df_scored):,} scored rows to {out_path}")
    return df_scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT -> Currency Mapper -> Batch FinBERT Pipeline")
    parser.add_argument("--days", type=int, default=5, help="Lookback days for live GDELT DOC 2.0 fetch")
    parser.add_argument("--chunk-hours", type=int, default=24, help="Window size (hours) per DOC API chunk")
    parser.add_argument("--input", type=str, default=None, help="Local historical GKG CSV/Parquet path")
    parser.add_argument("--historical", action="store_true", help="Query historical GDELT 2.0 GKG via BigQuery")
    parser.add_argument("--start-date", type=str, default=GKG_PAGE_TITLE_START_DATE, help=f"Start date (YYYY-MM-DD) for --historical (min {GKG_PAGE_TITLE_START_DATE})")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD) for --historical (default: tomorrow UTC)")
    parser.add_argument("--chunk-months", type=int, default=12, help="Months per BigQuery chunk for resumable backfill (0 = single query)")
    parser.add_argument("--since-last", action="store_true", help="Resume --historical from max published_utc in --output and merge")
    parser.add_argument("--dry-run", action="store_true", help="Estimate BigQuery scan size & cost ($0) without running")
    parser.add_argument("--max-gb-billed", type=float, default=None, help="Hard cumulative safety cap on total GiB billed across all BigQuery chunks")
    parser.add_argument("--gcp-project", type=str, default=None, help="GCP project ID for BigQuery (or GCP_PROJECT_ID in .env)")
    parser.add_argument("--save-raw-gkg", type=str, default=None, help="Optional Parquet path or directory to cache raw BigQuery export")
    parser.add_argument("--output", type=str, default="data/news_sentiment.parquet", help="Output Parquet cache path")
    parser.add_argument("--merge", action="store_true", help="Merge new scored rows into existing --output Parquet")
    parser.add_argument("--min-importance", type=float, default=0.25, help="Minimum importance threshold")
    parser.add_argument("--batch-size", type=int, default=128, help="FinBERT inference batch size")
    args = parser.parse_args()

    df_result = run_news_pipeline(
        days=args.days,
        chunk_hours=args.chunk_hours,
        input_gkg_path=args.input,
        historical=args.historical,
        start_date=args.start_date,
        end_date=args.end_date,
        chunk_months=args.chunk_months,
        since_last=args.since_last,
        dry_run=args.dry_run,
        max_gb_billed=args.max_gb_billed,
        gcp_project=args.gcp_project,
        save_raw_gkg=args.save_raw_gkg,
        output_path=args.output,
        merge=args.merge,
        min_importance=args.min_importance,
        batch_size=args.batch_size,
    )
    if not df_result.is_empty():
        print("\nSample scored rows:")
        print(df_result.select(["published_utc", "currency", "importance", "net_sentiment", "title"]).head(10))
