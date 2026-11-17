from __future__ import annotations

import re
import polars as pl

CURRENCY_LIST: list[str] = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# High-precision central bank, macro indicator, and currency keywords.
# Matched case-insensitively with word boundaries against title + organizations.
CURRENCY_PATTERNS: dict[str, list[str]] = {
    "USD": [
        r"\busd\b",
        r"\bu\.s\.?\s*dollar\b",
        r"\bgreenback\b",
        r"\bfederal reserve\b",
        r"\bthe fed\b",
        r"\bfed\b",
        r"\bfomc\b",
        r"\bpowell\b",
        r"\bnonfarm payrolls?\b",
        r"\bnfp\b",
        r"\bu\.?s\.?\s*(?:cpi|inflation|ppi|pce|gdp|retail sales|jobs|unemployment|treasury|yields?)\b",
        r"\btreasury yields?\b",
    ],
    "EUR": [
        r"\beur\b",
        r"\beuro\b",
        r"\beurozone\b",
        r"\beuro area\b",
        r"\becb\b",
        r"\beuropean central bank\b",
        r"\blagarde\b",
        r"\bgerman(?:y)?\s*(?:cpi|inflation|gdp|ifo|zew|pmi|bund)\b",
        r"\beurozone\s*(?:cpi|inflation|gdp|pmi)\b",
    ],
    "GBP": [
        r"\bgbp\b",
        r"\bpound\b",
        r"\bsterling\b",
        r"\bbank of england\b",
        r"\bboe\b",
        r"\bbailey\b",
        r"\bu\.?k\.?\s*(?:cpi|inflation|gdp|pmi|jobs|gilt| gilts|retail sales)\b",
        r"\bbritish\s*(?:pound|economy|inflation)\b",
    ],
    "JPY": [
        r"\bjpy\b",
        r"\byen\b",
        r"\bbank of japan\b",
        r"\bboj\b",
        r"\bueda\b",
        r"\bjapan(?:ese)?\s*(?:cpi|inflation|gdp|yen|yields?|jgb)\b",
        r"\btokyo\s*cpi\b",
    ],
    "CHF": [
        r"\bchf\b",
        r"\bswiss franc\b",
        r"\bswiss national bank\b",
        r"\bsnb\b",
        r"\bswitzerland\s*(?:cpi|inflation|gdp|rate)\b",
    ],
    "CAD": [
        r"\bcad\b",
        r"\bloonie\b",
        r"\bcanadian dollar\b",
        r"\bbank of canada\b",
        r"\bboc\b",
        r"\bmacklem\b",
        r"\bcanada\s*(?:cpi|inflation|gdp|jobs|employment)\b",
    ],
    "AUD": [
        r"\baud\b",
        r"\baussie\b",
        r"\baustralian dollar\b",
        r"\breserve bank of australia\b",
        r"\brba\b",
        r"\bbullock\b",
        r"\baustralia\s*(?:cpi|inflation|gdp|jobs|employment)\b",
    ],
    "NZD": [
        r"\bnzd\b",
        r"\bkiwi\b",
        r"\bnew zealand dollar\b",
        r"\breserve bank of new zealand\b",
        r"\brbnz\b",
        r"\bnew zealand\s*(?:cpi|inflation|gdp|jobs)\b",
    ],
}

# Explicit pair mentions like EURUSD, EUR/USD, GBP-JPY -> maps to both currencies.
PAIR_REGEX = re.compile(
    r"\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD)[/\-]?(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD)\b",
    re.IGNORECASE,
)

COMPILED_CURRENCY_REGEX: dict[str, re.Pattern[str]] = {
    ccy: re.compile("|".join(patterns), re.IGNORECASE)
    for ccy, patterns in CURRENCY_PATTERNS.items()
}

# Fallback country -> currency mapping using GDELT source_country / V2Locations
COUNTRY_TO_CURRENCY: dict[str, str] = {
    "united states": "USD",
    "us": "USD",
    "eurozone": "EUR",
    "germany": "EUR",
    "france": "EUR",
    "italy": "EUR",
    "spain": "EUR",
    "united kingdom": "GBP",
    "uk": "GBP",
    "japan": "JPY",
    "switzerland": "CHF",
    "canada": "CAD",
    "australia": "AUD",
    "new zealand": "NZD",
}

HIGH_IMPACT_PATTERN = re.compile(
    r"\b(interest rate|rate decision|rate hike|rate cut|basis points|bps|"
    r"cpi|inflation|pce|nfp|nonfarm payrolls?|gdp|unemployment|central bank|"
    r"fomc|ecb|boe|boj|snb|boc|rba|rbnz|powell|lagarde|bailey|ueda|"
    r"monetary policy|hawkish|dovish|recession|tariff|tariffs)\b",
    re.IGNORECASE,
)

MEDIUM_IMPACT_PATTERN = re.compile(
    r"\b(pmi|retail sales|trade balance|bond yield|treasury|forex|fx|"
    r"currency|exchange rate|dollar|euro|sterling|yen|franc|loonie|aussie|kiwi)\b",
    re.IGNORECASE,
)

HIGH_IMPACT_THEMES = ("ECON_CENTRALBANK", "ECON_INTEREST_RATES", "ECON_INFLATION")
MEDIUM_IMPACT_THEMES = ("ECON_CURRENCY", "ECON_GDP", "ECON_UNEMPLOYMENT")


def detect_currencies_and_importance(
    title: str,
    organizations: str = "",
    themes: str = "",
    locations: str = "",
    source_country: str = "",
) -> tuple[list[str], float]:
    """
    Identifies which of the 8 major FX currencies an article affects and
    returns (currencies, importance_score in [0.0, 1.0]).
    """
    text_blob = f"{title} {organizations}".strip()
    matched: set[str] = set()

    # 1. Explicit FX pair mentions (e.g., EUR/USD, USDJPY)
    for base, quote in PAIR_REGEX.findall(title):
        b, q = base.upper(), quote.upper()
        if b != q:
            matched.add(b)
            matched.add(q)

    # 2. Central bank / macro / currency keyword rules
    for ccy, regex in COMPILED_CURRENCY_REGEX.items():
        if regex.search(text_blob):
            matched.add(ccy)

    # 3. Importance scoring (text + GDELT V2Themes)
    importance = 0.0
    themes_upper = themes.upper()
    if HIGH_IMPACT_PATTERN.search(text_blob) or any(t in themes_upper for t in HIGH_IMPACT_THEMES):
        importance = 1.0
    elif MEDIUM_IMPACT_PATTERN.search(text_blob) or any(t in themes_upper for t in MEDIUM_IMPACT_THEMES):
        importance = 0.5
    elif matched:
        importance = 0.25

    # 4. Country fallback only when macro-relevant but no explicit currency keyword matched
    if not matched and importance >= 0.5:
        loc_lower = f"{source_country} {locations}".lower()
        for country_name, ccy in COUNTRY_TO_CURRENCY.items():
            if country_name in loc_lower:
                matched.add(ccy)

    return sorted(matched), importance


def map_news_to_currencies(
    df_news: pl.DataFrame,
    min_importance: float = 0.25,
) -> pl.DataFrame:
    """
    Filters a normalized news DataFrame to FX-relevant items and explodes
    rows by currency so each row maps cleanly to a single currency in CURRENCY_LIST.
    """
    if df_news.is_empty():
        return pl.DataFrame(
            schema={
                "published_utc": pl.Datetime("us", time_zone="UTC"),
                "currency": pl.String,
                "importance": pl.Float32,
                "title": pl.String,
                "url": pl.String,
                "domain": pl.String,
            }
        )

    out_rows: list[dict] = []
    for row in df_news.iter_rows(named=True):
        currencies, importance = detect_currencies_and_importance(
            title=row.get("title") or "",
            organizations=row.get("organizations") or "",
            themes=row.get("themes") or "",
            locations=row.get("locations") or "",
            source_country=row.get("source_country") or "",
        )
        if not currencies or importance < min_importance:
            continue

        for ccy in currencies:
            out_rows.append(
                {
                    "published_utc": row["published_utc"],
                    "currency": ccy,
                    "importance": float(importance),
                    "title": row["title"],
                    "url": row.get("url") or "",
                    "domain": row.get("domain") or "",
                }
            )

    if not out_rows:
        return pl.DataFrame(
            schema={
                "published_utc": pl.Datetime("us", time_zone="UTC"),
                "currency": pl.String,
                "importance": pl.Float32,
                "title": pl.String,
                "url": pl.String,
                "domain": pl.String,
            }
        )

    return (
        pl.DataFrame(out_rows)
        .with_columns(
            [
                pl.col("published_utc").cast(pl.Datetime("us", time_zone="UTC")),
                pl.col("importance").cast(pl.Float32),
            ]
        )
        .sort(["published_utc", "currency"])
    )
