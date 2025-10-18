import numpy as np
import torch
import pandas as pd
from pathlib import Path

from getting_data_service import get_last_month_by_symbol
from neural_network import StockPriceNeuralNetwork
from processing_data_service import preprocessing_dataframe, preprocessing_dataframe_non_usd, \
    denormalize_predicted_value
from show_candlestick_chart_ui import show_dataframe_and_prediction

#dataset path
try:
    import kagglehub
    PATH = kagglehub.dataset_download("mattiuzc/stock-exchange-data")
except:
    print("Unable to get training dataset, trying cache...")
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
if not model_loaded:
    df = pd.read_csv(PATH)
    train_X, train_Y, test_X, test_Y = preprocessing_dataframe_non_usd(df, device=DEVICE)
    train_set = torch.utils.data.TensorDataset(train_X, train_Y)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=DEVICE))
    test_set = torch.utils.data.TensorDataset(test_X, test_Y)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=DEVICE))

    #Training
    for t in range(N_EPOCHS):
        print(f"Epoch {t}\n-------------------------------")
        MODEL.train_loop(train_dataloader, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
        MODEL.test_loop(test_dataloader)
    torch.save(MODEL.state_dict(), 'model.obj')
    print("Neural Network training session saved")
else:
    print("Neural Network loaded")

stock_symbol = input("Stock symbol (to predict): ")
stock_df = get_last_month_by_symbol(stock_symbol)
input_tensor = preprocessing_dataframe(stock_df, device=DEVICE, for_model_training=False)
next_price = MODEL(MODEL.flatten(input_tensor))
next_price = torch.squeeze(next_price).to("cpu").detach()
next_price = np.float32(next_price)
next_price = denormalize_predicted_value(next_price, stock_df)
show_dataframe_and_prediction(stock_df, float(next_price))
