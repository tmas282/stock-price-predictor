import pandas as pd
import numpy as np
import torch


def preprocessing_dataframe(df: pd.DataFrame, device:str = "cpu", for_model_training=True):
    """
    This function task is to convert and normalize for error pruning on the neural network
    :param for_model_training: If this dataframe is for model training
    :param df: must have Date, Open, High, Low, Close, Volume (in USD)
    :param device: the torch.accelerator device, if None is specified, cpu is the default
    :return: four tensors located on the device, being two (X, Y) for training and more two (X, Y) for validation, or only one Tensor to use the model
    """
    df = df.copy()
    df["CloseUSD"] = df["Close"]
    return preprocessing_dataframe_non_usd(df, device, for_model_training)

def preprocessing_dataframe_non_usd(df: pd.DataFrame, device:str = "cpu", for_model_training=True):
    """
    This function task is to convert and normalize for error pruning on the neural network
    :param for_model_training: If this dataframe is for model training
    :param df: must have Date, Open, High, Low, Close, Volume, CloseUSD
    :param device: the torch.accelerator device, if None is specified, cpu is the default
    :return: four tensors located on the device, being two (X, Y) for training and more two (X, Y) for validation, or only one Tensor to use the model
    """
    #Preprocessing
    #Params shape n * 30 days * 6 columns on csv
    #Y: the close value of next month: n+1 * first day
    df = df.copy()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "CloseUSD"]]
    #Date to integer
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df.Date.dt.strftime('%Y%m%d').astype(int)
    #Values to USD
    df["Open"] = df["Open"] * df["CloseUSD"] / df["Close"]
    df["High"] = df["High"] * df["CloseUSD"] / df["Close"]
    df["Low"] = df["Low"] * df["CloseUSD"] / df["Close"]
    df["Close"] = df["CloseUSD"]
    #Values normalized between 0 and 1
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Open"] = (df["Open"] - df["Open"].min()) / (df["Open"].max() - df["Open"].min())
    df["High"] = (df["High"] - df["High"].min()) / (df["High"].max() - df["High"].min())
    df["Low"] = (df["Low"] - df["Low"].min()) / (df["Low"].max() - df["Low"].min())
    df["Close"] = (df["Close"] - df["Close"].min()) / (df["Close"].max() - df["Close"].min())
    df["Volume"] = (df["Volume"] - df["Volume"].min()) / (df["Volume"].max() - df["Volume"].min())
    if(for_model_training):
        split_time_series = np.split(df, list(range(0, len(df), 30)))
        split_time_series.pop(0)
        uncomplete_month = split_time_series.pop()

        #Spliting data into testing and validation
        split_breakpoint = round(len(split_time_series) * 0.8)

        train_X = np.array(split_time_series[0:(split_breakpoint-1)])
        train_Y = np.ndarray(shape=(len(train_X)))
        m=0
        for v in split_time_series[1:split_breakpoint]:
            train_Y[m] = v["Close"].tolist()[0]
            m+=1

        test_X = np.array(split_time_series[split_breakpoint:])
        test_Y = np.ndarray(shape=(len(test_X)))
        m=0
        for v in split_time_series[split_breakpoint+1:]:
            test_Y[m] = v["Close"].tolist()[0]
            m += 1
        test_Y[m] = uncomplete_month["Close"].tolist()[0]

        train_X = torch.from_numpy(train_X).to(dtype=torch.float32).to(device)
        train_Y = torch.from_numpy(train_Y).to(dtype=torch.float32).to(device)
        test_X = torch.from_numpy(test_X).to(dtype=torch.float32).to(device)
        test_Y = torch.from_numpy(test_Y).to(dtype=torch.float32).to(device)
        return train_X, train_Y, test_X, test_Y
    else:
        return torch.from_numpy(np.reshape(np.array(df), shape=(1, 30, 6))).to(dtype=torch.float32).to(device)