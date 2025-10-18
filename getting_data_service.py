from dotenv import load_dotenv
import requests, os, pandas as pd, numpy as np

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def get_last_month_by_symbol(symbol: str) -> pd.DataFrame:
    """
    This gets the last 30 days of stock Date, Open, High, Low, CLose, Volume in USD using alphavantage API.
    :param symbol: the stock Symbol
    :return: pd.DataFrame with Date, Open, High, Low, CLose, Volume in USD
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    time_series = data['Time Series (Daily)']
    dates = sorted(time_series.keys(), reverse=True)[:30]
    dates = sorted(dates)
    df = pd.DataFrame()
    df["Date"] = dates
    for i, date in enumerate(dates):
        df.loc[i, "Open"] = np.float32(time_series[date]["1. open"])
        df.loc[i, "High"] = np.float32(time_series[date]["2. high"])
        df.loc[i, "Low"] = np.float32(time_series[date]["3. low"])
        df.loc[i, "Close"] = np.float32(time_series[date]["4. close"])
        df.loc[i, "Volume"] = np.float32(time_series[date]["5. volume"])
    return df