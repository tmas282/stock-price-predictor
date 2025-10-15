import kagglehub
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import StockPriceNeuralNetwork, train_loop, test_loop

#Rules
N_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.02
PATH = kagglehub.dataset_download("mattiuzc/stock-exchange-data")
PATH += "/indexProcessed.csv"
# DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {DEVICE} device")
# torch.set_default_device(DEVICE)
# torch.set_default_dtype(torch.float32)

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
m=0
for v in split_time_series[1:split_breakpoint]:
    train_Y[m] = v["Adj Close"].tolist()[0]
    m+=1

test_X = np.array(split_time_series[split_breakpoint:])
test_Y = np.ndarray(shape=(len(test_X)))
m=0
for v in split_time_series[split_breakpoint+1:]:
    test_Y[m] = v["Adj Close"].tolist()[0]
    m += 1
test_Y[m] = uncomplete_month["Adj Close"].tolist()[0]

print(f"Dataframe shape: {np.shape(split_time_series)}"
      f"\nTrain shape: {np.shape(train_X)} Validate shape: {np.shape(test_X)}"
      f"\nTrain Y shape: {np.shape(train_Y)} Validate Y shape: {np.shape(test_Y)}")

train_X = torch.from_numpy(train_X).to(dtype=torch.float32)#.to(DEVICE)
train_Y = torch.from_numpy(train_Y).to(dtype=torch.float32)#.to(DEVICE)
test_X = torch.from_numpy(test_X).to(dtype=torch.float32)#.to(DEVICE)
test_Y = torch.from_numpy(test_Y).to(dtype=torch.float32)#.to(DEVICE)

trainset = torch.utils.data.TensorDataset(train_X, train_Y)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)#, generator=torch.Generator(device=DEVICE))
testset = torch.utils.data.TensorDataset(test_X, test_Y)
testloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)#, generator=torch.Generator(device=DEVICE))

#Model
model = StockPriceNeuralNetwork()
for t in range(N_EPOCHS):
    print(f"Epoch {t}\n-------------------------------")
    train_loop(trainloader, model, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    test_loop(testloader, model)

