# EUR/USD Trading ML System Solutions Study

## News Analysis
- FinBERT: Evaluation of news regarding macro-economics, inflation, etc. Must be for EU market and US market.
- Potential Sources:
  1. FXStreet News Feed
  2. Reuters RSS / Markets Feed
  3. Dow Jones Newswires / Bloomberg Terminal API
  4. FOMC Meeting Minutes & Statement Texts
  5. ECB Press Conference Transcripts

## Timeframes & Indicators
- 1D:
  - SMA (50, 200)
  - VMA
  - Volume
  - ATR
- 4H:
  - EMA (10,20)
  - RSI
  - MACD
  - Volume
  - Bollinger Bands
- 1H:
  - Volume Profile Visible Range
  - Volume
  - VWAP
  - Session Timers

### Observations
> Data is non-stationary. Data follows a mean, median, outliers according to the interval.
>
> Lookback for 1D is between 1 year. For 4H is 3 months. For 1H is 2 weeks.

## Model Output
- Next Candlestick (Autoregression task)

## Model Validation and Test
- Model needs to be trained, validated and tested sequentially, data cannot be randomly split. E.g.: Model was trained with 2010-2023, must be validated with 2023-2025, tested with 2025-2026.


## Tech Stack
- Pytorch
- Polars

## Dataset
https://huggingface.co/datasets/Yllvar/fx-historical-data


## Background training

```bash
nohup bash -c "uv run main.py" > output.log 2>&1 &
```
