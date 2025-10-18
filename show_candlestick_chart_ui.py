import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

def show_dataframe_and_prediction(df: pd.DataFrame, prediction: float, stock_symbol=None):
    """
    Shows the DataFrame and the prediction in a candlestick plot
    :param prediction: the prediction
    :param stock_symbol: the stock symbol
    :param df: the dataframe with values, cols must contain ['Date', 'Open', 'High', 'Low', 'Close']
    :return: None
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
    last_date = df.index[-1]
    tomorrow = last_date + pd.Timedelta(days=1)
    title = f'Candlestick Chart {f"of {stock_symbol}" if stock_symbol is not None else ""}'
    fig, axlist = mpf.plot(
        df,
        type='candle',
        title=title,
        style='yahoo',
        returnfig=True
    )
    ax = axlist[0]
    x_pos = len(df.index)
    ax.scatter(x_pos, prediction, s=100, marker='o', color='blue')
    ax.set_xticks(range(len(df.index) + 1))
    xlabels = []
    for v in list(df.index) + [tomorrow]:
        xlabels.append(pd.Timestamp(v).strftime('%d-%m-%Y'))
    ax.set_xticklabels(xlabels, rotation=45)
    plt.show()