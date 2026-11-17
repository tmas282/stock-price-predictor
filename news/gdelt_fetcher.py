from __future__ import annotations

import html
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
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

# BigQuery SQL template for GDELT 2.0 GKG (Feb 2015 -> present).
# Extracts <PAGE_TITLE> from Extras and filters to macro/central-bank themes.
HISTORICAL_GKG_SQL = """
SELECT
  PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS published_utc,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
  DocumentIdentifier AS url,
  SourceCommonName AS domain,
  '' AS source_country,
  V2Themes AS themes,
  V2Organizations AS organizations,
  V2Locations AS locations
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{start_date}')
  AND _PARTITIONTIME < TIMESTAMP('{end_date}')
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


def _query_gdelt_doc(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    max_records: int = 250,
    timeout: int = 30,
) -> list[dict]:
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    if not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return payload.get("articles", []) or []


def fetch_gdelt_last_n_days(
    days: int = 5,
    queries: list[str] | None = None,
    max_records_per_query: int = 250,
    end_dt: datetime | None = None,
    sleep_seconds: float = 0.5,
) -> pl.DataFrame:
    """
    Fetches recent FX/macro news headlines from the free GDELT DOC 2.0 API
    over the last `days` days (default: 5 days) and returns a deduplicated
    Polars DataFrame with UTC timestamps.
    """
    if queries is None:
        queries = DEFAULT_FX_QUERIES
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    rows: list[dict] = []
    for idx, q in enumerate(queries):
        try:
            articles = _query_gdelt_doc(
                query=q,
                start_dt=start_dt,
                end_dt=end_dt,
                max_records=max_records_per_query,
            )
        except Exception as exc:
            print(f"[GDELT DOC] Query {idx + 1}/{len(queries)} failed: {exc}")
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

        if idx < len(queries) - 1 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not rows:
        return _empty_news_df()

    df = pl.DataFrame(rows, schema=NEWS_SCHEMA)
    # Deduplicate by URL first, then by normalized title + hour bucket
    df = (
        df.unique(subset=["url"], keep="first")
        .with_columns(pl.col("title").str.to_lowercase().alias("_title_norm"))
        .unique(subset=["_title_norm"], keep="first")
        .drop("_title_norm")
        .sort("published_utc")
    )
    return df


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

    # Ensure all expected columns exist
    for col_name, dtype in NEWS_SCHEMA.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    if df["published_utc"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(
            pl.col("published_utc").str.to_datetime(time_zone="UTC")
        )
    elif df["published_utc"].dtype == pl.Datetime("us") or df["published_utc"].dtype == pl.Datetime("ns"):
        df = df.with_columns(
            pl.col("published_utc").dt.replace_time_zone("UTC").cast(pl.Datetime("us", time_zone="UTC"))
        )
    else:
        df = df.with_columns(
            pl.col("published_utc").cast(pl.Datetime("us", time_zone="UTC"))
        )

    df = (
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
    return df


def fetch_historical_bigquery(
    start_date: str = "2015-02-18",
    end_date: str | None = None,
    project_id: str | None = None,
    output_path: str | Path | None = None,
) -> pl.DataFrame:
    """
    Optional helper to query GDELT 2.0 GKG directly via google-cloud-bigquery.
    Requires `google-cloud-bigquery` and GCP credentials configured.
    """
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-cloud-bigquery is not installed. Either install it "
            "(`uv add google-cloud-bigquery`) or run HISTORICAL_GKG_SQL in "
            "the BigQuery console and load the file with load_gkg_export()."
        ) from exc

    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sql = HISTORICAL_GKG_SQL.format(start_date=start_date, end_date=end_date)
    client = bigquery.Client(project=project_id)
    arrow_table = client.query(sql).to_arrow()
    df = pl.from_arrow(arrow_table)
    if isinstance(df, pl.Series):
        df = df.to_frame()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)

    return df
