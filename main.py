import kagglehub
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame

EPOCHS = 10
PATH = kagglehub.dataset_download("mattiuzc/stock-exchange-data")
PATH += "/indexProcessed.csv"

#Preprocessing
#Params shape n * 30 days * 9 columns on csv
#Y: the adj close value of next month: n+1 * first day

df = pd.read_csv(PATH)
df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
df["Date"] = pd.to_datetime(df["Date"])
df["Date"] = df.Date.dt.strftime('%Y%m%d').astype(int)
split_time_series = np.split(df, list(range(0, len(df), 30)))
split_time_series.pop(0)
uncomplete_month = split_time_series.pop()

#Spliting data into testing and validation
split_breakpoint = round(len(split_time_series) * 0.8)

train_X = np.array(split_time_series[0:(split_breakpoint-1)])
train_Y = np.ndarray(shape=(len(train_X)))
for v in split_time_series[1:split_breakpoint]:
    np.append(train_Y, v["Adj Close"].tolist()[0])

test_X = np.array(split_time_series[split_breakpoint:])
test_Y = np.ndarray(shape=(len(test_X)))
for v in split_time_series[split_breakpoint+1:]:
    np.append(test_Y, v["Adj Close"].tolist()[0])
np.append(test_Y, uncomplete_month["Adj Close"].tolist()[0])

print(f"Dataframe shape: {np.shape(split_time_series)}"
      f"\nTrain shape: {np.shape(train_X)} Validate shape: {np.shape(test_X)}"
      f"\nTrain Y shape: {np.shape(train_Y)} Validate Y shape: {np.shape(test_Y)}")

train_X = torch.from_numpy(train_X)
train_Y = torch.from_numpy(train_Y)
test_X = torch.from_numpy(test_X)
test_Y = torch.from_numpy(test_Y)

#Model
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

