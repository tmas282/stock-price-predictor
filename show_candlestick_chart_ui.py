import mplfinance as mpf
import pandas as pd

def show_dataframe_and_prediction(df: pd.DataFrame, stock_symbol=None):
    """
    Shows the DataFrame and the prediction in a candlestick plot
    :param stock_symbol: the stock symbol
    :param df: the dataframe with values, cols must contain ['Date', 'Open', 'High', 'Low', 'Close']
    :return: None
    """
    df['Date'] = pd.to_datetime(df['Date'])
    title = f'Candlestick Chart of {f"of {stock_symbol}" if not stock_symbol is None else ""}'
    df.set_index('Date', inplace=True)
    mpf.plot(df, type='candle', title=title, style='yahoo')