from __future__ import annotations

import html
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import polars as pl

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Targeted macro & central-bank queries so the DOC API returns FX-relevant articles
# rather than general world news.
DEFAULT_FX_QUERIES: list[str] = [
    '("Federal Reserve" OR FOMC OR "US inflation" OR "US CPI" OR "nonfarm payrolls" OR "US GDP" OR "US dollar") sourcelang:eng',
    '("European Central Bank" OR ECB OR "Eurozone inflation" OR "Eurozone CPI" OR "German CPI" OR euro) sourcelang:eng',
    '("Bank of England" OR BoE OR "UK inflation" OR "UK CPI" OR "UK GDP" OR "British pound" OR sterling) sourcelang:eng',
    '("Bank of Japan" OR BoJ OR "Japanese yen" OR "Japan CPI" OR "Tokyo CPI" OR yen) sourcelang:eng',
    '("Swiss National Bank" OR SNB OR "Bank of Canada" OR BoC OR "Reserve Bank of Australia" OR RBA OR "RBNZ") sourcelang:eng',
    '(EURUSD OR GBPUSD OR USDJPY OR forex OR "interest rate" OR "central bank" OR "monetary policy") sourcelang:eng',
]

# Earliest date GDELT 2.0 GKG populated <PAGE_TITLE> inside Extras XML.
# Partitions before this date contain 0 <PAGE_TITLE> rows.
GKG_PAGE_TITLE_START_DATE = "2019-09-22"

# BigQuery SQL template for GDELT 2.0 GKG (2019-09-22 -> present).
# Extracts <PAGE_TITLE> from Extras, filters to native English macro/central-bank
# articles, and omits heavy V2Organizations/V2Locations columns to minimize bytes scanned.
HISTORICAL_GKG_SQL = """
SELECT
  PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS published_utc,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
  DocumentIdentifier AS url,
  SourceCommonName AS domain,
  '' AS source_country,
  V2Themes AS themes,
  '' AS organizations,
  '' AS locations
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{start_date}')
  AND _PARTITIONTIME < TIMESTAMP('{end_date}')
  AND (TranslationInfo IS NULL OR TranslationInfo = '')
  AND Extras LIKE '%<PAGE_TITLE>%'
  AND (
    V2Themes LIKE '%ECON_CENTRALBANK%'
    OR V2Themes LIKE '%ECON_INFLATION%'
    OR V2Themes LIKE '%ECON_INTEREST_RATES%'
    OR V2Themes LIKE '%ECON_CURRENCY%'
    OR V2Themes LIKE '%ECON_UNEMPLOYMENT%'
    OR V2Themes LIKE '%ECON_GDP%'
  )
"""

NEWS_SCHEMA = {
    "published_utc": pl.Datetime("us", time_zone="UTC"),
    "title": pl.String,
    "url": pl.String,
    "domain": pl.String,
    "source_country": pl.String,
    "themes": pl.String,
    "organizations": pl.String,
    "locations": pl.String,
}


def _empty_news_df() -> pl.DataFrame:
    return pl.DataFrame(schema=NEWS_SCHEMA)


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(html.unescape(str(text)).split())


def _parse_gdelt_seendate(seendate: str | None) -> datetime | None:
    if not seendate:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(seendate, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


def _compute_backoff_delay(
    attempt: int,
    base_backoff: float,
    headers: object | None = None,
) -> float:
    """Computes retry delay using Retry-After when valid, else exponential backoff with jitter."""
    retry_after = headers.get("Retry-After") if headers is not None and hasattr(headers, "get") else None # pyright: ignore[reportAttributeAccessIssue]
    if retry_after is not None:
        raw_value = str(retry_after).strip()
        try:
            seconds = float(raw_value)
            if seconds > 0:
                return seconds
        except ValueError:
            try:
                retry_dt = parsedate_to_datetime(raw_value)
                if retry_dt.tzinfo is None:
                    retry_dt = retry_dt.replace(tzinfo=timezone.utc)
                seconds = (retry_dt - datetime.now(timezone.utc)).total_seconds()
                if seconds > 0:
                    return seconds
            except (TypeError, ValueError):
                pass
    return base_backoff * (2 ** attempt) + random.uniform(0.0, 1.5)


def _query_gdelt_doc(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    max_records: int = 250,
    timeout: int = 30,
    max_retries: int = 4,
    base_backoff: float = 5.0,
) -> list[dict]:
    """Queries the GDELT DOC 2.0 ArtList endpoint with backoff on 429/5xx and rate-limit text."""
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "maxrecords": str(max_records),
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }
    url = f"{GDELT_DOC_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "stock-price-predictor/1.0 (research)"},
    )

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                resp_headers = getattr(resp, "headers", None)
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_STATUS or attempt >= max_retries:
                raise
            delay = _compute_backoff_delay(attempt, base_backoff, exc.headers)
            exc.close()
            time.sleep(delay)
            continue
        except (urllib.error.URLError, TimeoutError):
            if attempt >= max_retries:
                raise
            time.sleep(_compute_backoff_delay(attempt, base_backoff))
            continue

        if not raw.strip():
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raw_lower = raw.lower()
            if any(tok in raw_lower for tok in ("limit", "rate", "too many")):
                if attempt < max_retries:
                    time.sleep(_compute_backoff_delay(attempt, base_backoff, resp_headers))
                    continue
                raise RuntimeError(f"GDELT DOC rate limit response: {raw.strip()[:200]}")
            return []
        return payload.get("articles", []) or []

    return []


def fetch_gdelt_last_n_days(
    days: int = 5,
    queries: list[str] | None = None,
    max_records_per_query: int = 250,
    end_dt: datetime | None = None,
    sleep_seconds: float = 5.5,
    chunk_hours: int = 24,
) -> pl.DataFrame:
    """
    Fetches recent FX/macro news headlines from the free GDELT DOC 2.0 API
    over the last `days` days (default: 5 days), chunked into `chunk_hours`
    windows so GDELT's 250-record cap per query doesn't truncate older days.
    """
    if queries is None:
        queries = DEFAULT_FX_QUERIES
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    # Build chronological time chunks of `chunk_hours`
    chunks: list[tuple[datetime, datetime]] = []
    cur = start_dt
    step = timedelta(hours=max(1, chunk_hours))
    while cur < end_dt:
        nxt = min(cur + step, end_dt)
        chunks.append((cur, nxt))
        cur = nxt

    total_calls = len(chunks) * len(queries)
    call_idx = 0
    rows: list[dict] = []

    for c_start, c_end in chunks:
        for q_idx, q in enumerate(queries):
            call_idx += 1
            try:
                articles = _query_gdelt_doc(
                    query=q,
                    start_dt=c_start,
                    end_dt=c_end,
                    max_records=max_records_per_query,
                )
            except Exception as exc:
                print(
                    f"[GDELT DOC] Call {call_idx}/{total_calls} "
                    f"({c_start:%Y-%m-%d %H:%M}..{c_end:%Y-%m-%d %H:%M}, q={q_idx + 1}) failed: {exc}"
                )
                articles = []

            for art in articles:
                title = _clean_text(art.get("title"))
                published = _parse_gdelt_seendate(art.get("seendate"))
                if not title or published is None:
                    continue
                rows.append(
                    {
                        "published_utc": published,
                        "title": title,
                        "url": str(art.get("url") or ""),
                        "domain": str(art.get("domain") or ""),
                        "source_country": str(art.get("sourcecountry") or ""),
                        "themes": "",
                        "organizations": "",
                        "locations": "",
                    }
                )

            if call_idx < total_calls and sleep_seconds > 0:
                time.sleep(sleep_seconds)

    if not rows:
        return _empty_news_df()

    df = pl.DataFrame(rows, schema=NEWS_SCHEMA)
    # Deduplicate by URL first, then by normalized title
    df = (
        df.unique(subset=["url"], keep="first")
        .with_columns(pl.col("title").str.to_lowercase().alias("_title_norm"))
        .unique(subset=["_title_norm"], keep="first")
        .drop("_title_norm")
        .sort("published_utc")
    )
    return df


def _normalize_news_df(df: pl.DataFrame) -> pl.DataFrame:
    """Normalizes a raw GKG DataFrame (from CSV/Parquet or BigQuery) to NEWS_SCHEMA."""
    if df.is_empty():
        return _empty_news_df()

    for col_name, dtype in NEWS_SCHEMA.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    if df["published_utc"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(
            pl.col("published_utc").str.to_datetime(time_zone="UTC")
        )
    elif df["published_utc"].dtype in (pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("ms")):
        df = df.with_columns(
            pl.col("published_utc").dt.replace_time_zone("UTC").cast(pl.Datetime("us", time_zone="UTC"))
        )
    else:
        df = df.with_columns(
            pl.col("published_utc").cast(pl.Datetime("us", time_zone="UTC"))
        )

    return (
        df.select(list(NEWS_SCHEMA.keys()))
        .with_columns(
            [
                pl.col("title").map_elements(_clean_text, return_dtype=pl.String),
                pl.col("url").fill_null(""),
                pl.col("domain").fill_null(""),
                pl.col("source_country").fill_null(""),
                pl.col("themes").fill_null(""),
                pl.col("organizations").fill_null(""),
                pl.col("locations").fill_null(""),
            ]
        )
        .filter(pl.col("title").str.len_chars() > 0)
        .unique(subset=["url"], keep="first")
        .sort("published_utc")
    )


def load_gkg_export(path: str | Path) -> pl.DataFrame:
    """
    Loads a historical GDELT 2.0 GKG export (Parquet or CSV produced by
    HISTORICAL_GKG_SQL) and normalizes it to NEWS_SCHEMA.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GKG export not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format '{path.suffix}'. Use .parquet or .csv.")

    return _normalize_news_df(df)


def clamp_gkg_start_date(start_date: str) -> str:
    """Clamps start_date to GKG_PAGE_TITLE_START_DATE (2019-09-22) with a warning."""
    if start_date < GKG_PAGE_TITLE_START_DATE:
        print(
            f"[BigQuery] Warning: GDELT GKG <PAGE_TITLE> only exists from {GKG_PAGE_TITLE_START_DATE} onward. "
            f"Clamping start_date {start_date} -> {GKG_PAGE_TITLE_START_DATE} to avoid scanning 0-row partitions."
        )
        return GKG_PAGE_TITLE_START_DATE
    return start_date


def estimate_historical_bigquery_cost(
    start_date: str = GKG_PAGE_TITLE_START_DATE,
    end_date: str | None = None,
    project_id: str | None = None,
) -> dict[str, float | int]:
    """
    Runs a free BigQuery dry-run for HISTORICAL_GKG_SQL and returns exact
    bytes processed and estimated on-demand USD cost ($6.25/TiB, first 1 TiB/mo free).
    """
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-cloud-bigquery is not installed. Run: uv add google-cloud-bigquery pyarrow"
        ) from exc

    start_date = clamp_gkg_start_date(start_date)
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sql = HISTORICAL_GKG_SQL.format(start_date=start_date, end_date=end_date)
    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(sql, job_config=job_config)
    total_bytes = int(job.total_bytes_processed or 0)
    gib = total_bytes / (1024 ** 3)
    tib = total_bytes / (1024 ** 4)
    est_usd = tib * 6.25
    return {
        "bytes": total_bytes,
        "gib": gib,
        "tib": tib,
        "est_usd": est_usd,
    }


def fetch_historical_bigquery(
    start_date: str = GKG_PAGE_TITLE_START_DATE,
    end_date: str | None = None,
    project_id: str | None = None,
    output_path: str | Path | None = None,
    max_gb_billed: float | None = None,
) -> pl.DataFrame:
    """
    Queries GDELT 2.0 GKG directly via google-cloud-bigquery (2019-09-22 -> present)
    and normalizes the result to NEWS_SCHEMA.
    Requires `google-cloud-bigquery`, `pyarrow`, and GCP credentials configured.
    """
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-cloud-bigquery is not installed. Run: uv add google-cloud-bigquery pyarrow"
        ) from exc

    start_date = clamp_gkg_start_date(start_date)
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sql = HISTORICAL_GKG_SQL.format(start_date=start_date, end_date=end_date)
    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig()
    if max_gb_billed is not None and max_gb_billed > 0:
        job_config.maximum_bytes_billed = int(max_gb_billed * (1024 ** 3))

    query_job = client.query(sql, job_config=job_config)
    try:
        arrow_table = query_job.to_arrow()
    except Exception as exc:
        if "readsessions.create" in str(exc) or "PermissionDenied" in type(exc).__name__:
            print(
                "[BigQuery] No bigquery.readsessions.create permission; "
                "falling back to REST download (grant roles/bigquery.readSessionUser for faster downloads)."
            )
            arrow_table = query_job.to_arrow(create_bqstorage_client=False)
        else:
            raise
    df = pl.from_arrow(arrow_table)
    if isinstance(df, pl.Series):
        df = df.to_frame()

    df = _normalize_news_df(df)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)

    return df
