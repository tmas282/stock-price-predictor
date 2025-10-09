import kagglehub
import tensorflow as tf
import pandas as pd
import numpy as np

path = kagglehub.dataset_download("mattiuzc/stock-exchange-data")
path += "/indexProcessed.csv"

#Preprocessing
#Params shape 30 days * 9 columns on .csv
#Regression on Y: the adj close value

df = pd.read_csv(path)


#Model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(shape=(30, 9)),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.MeanSquaredError,
    metrics=["accuracy"]
)

#model.fit(epochs=10, x=[], y=[])
#model.evaluate(epochs=10, x=[], y=[])