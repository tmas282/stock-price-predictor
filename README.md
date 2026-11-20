# Trading ML System

## Background training

```bash
nohup bash -c "uv run main.py" > output.log 2>&1 &
```

## News Sentiment Retrieval

```bash
uv run .\news_pipeline.py --historical --start-date 2019-09-22 --dry-run
```
```bash
uv run .\news_pipeline.py --historical --start-date 2019-09-22 --chunk-months 12 --max-gb-billed 400 --output data/news_sentiment.parquet
```
```bash
uv run .\news_pipeline.py --historical --since-last --chunk-months 12 --max-gb-billed 400 --output data/news_sentiment.parquet
```