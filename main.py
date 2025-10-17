import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from NeuralNetwork import StockPriceNeuralNetwork
#dataset path
try:
    import kagglehub
    PATH = kagglehub.dataset_download("mattiuzc/stock-exchange-data")
except:
    print("Unable to get dataset, trying cache...")
    PATH = f"{Path.home()}/.cache/kagglehub/datasets/mattiuzc/stock-exchange-data/versions/2"
#Rules
N_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.002
PATH += "/indexProcessed.csv"
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {DEVICE} device")
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float32)
MODEL = StockPriceNeuralNetwork()
model_loaded = True
try:
    MODEL.load_state_dict(torch.load('model.obj', weights_only=True))
except:
    model_loaded = False

#Preprocessing
#Params shape n * 30 days * 6 columns on csv
#Y: the close value of next month: n+1 * first day
df = pd.read_csv(PATH)
df = df[["Date", "Open", "High", "Low", "Close", "Volume", "CloseUSD"]]
df["Date"] = pd.to_datetime(df["Date"])
df["Date"] = df.Date.dt.strftime('%Y%m%d').astype(int)
df["Open"] = df["Open"] * df["CloseUSD"] / df["Close"]
df["High"] = df["High"] * df["CloseUSD"] / df["Close"]
df["Low"] = df["Low"] * df["CloseUSD"] / df["Close"]
df["Close"] = df["CloseUSD"]
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
df["Open"] = (df["Open"] - df["Open"].min()) / (df["Open"].max() - df["Open"].min())
df["High"] = (df["High"] - df["High"].min()) / (df["High"].max() - df["High"].min())
df["Low"] = (df["Low"] - df["Low"].min()) / (df["Low"].max() - df["Low"].min())
df["Close"] = (df["Close"] - df["Close"].min()) / (df["Close"].max() - df["Close"].min())
df["Volume"] = (df["Volume"] - df["Volume"].min()) / (df["Volume"].max() - df["Volume"].min())
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

train_X = torch.from_numpy(train_X).to(dtype=torch.float32).to(DEVICE)
train_Y = torch.from_numpy(train_Y).to(dtype=torch.float32).to(DEVICE)
test_X = torch.from_numpy(test_X).to(dtype=torch.float32).to(DEVICE)
test_Y = torch.from_numpy(test_Y).to(dtype=torch.float32).to(DEVICE)

train_set = torch.utils.data.TensorDataset(train_X, train_Y)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=DEVICE))
test_set = torch.utils.data.TensorDataset(test_X, test_Y)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=DEVICE))

#Training
if not model_loaded:
    for t in range(N_EPOCHS):
        print(f"Epoch {t}\n-------------------------------")
        MODEL.train_loop(train_dataloader, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
        MODEL.test_loop(test_dataloader)
    torch.save(MODEL.state_dict(), 'model.obj')
    print("Neural Network training session saved")
else:
    print("Neural Network loaded")

