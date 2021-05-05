from vrae.vrae import VRAE
from vrae.utils import *
import numpy as np
import torch
import pandas as pd
import plotly
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

dload = './model_dir'
hidden_size = 90
hidden_layer_depth = 2
latent_length = 20
batch_size = 32
learning_rate = 0.0005
n_epochs = 40
dropout_rate = 0.2
optimizer = 'Adam' # options: ADAM, SGD
cuda = False # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options: LSTM, GRU
data = pd.read_pickle("./walk_data.pkl")
data = data[:2000]
#
def windowed_dataset(y, input_window=5, output_window=1, stride=1, num_features=data.shape[1]):
    '''
    create a windowed dataset

    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model
    : param output_window:    number of future y samples to predict
    : param stide:            spacing between windows
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''

    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1

    X = np.zeros([input_window, num_samples, num_features])
    # Y = np.zeros([output_window, num_samples, num_features])

    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = ii #stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]

            # start_y = stride * ii + input_window
            # end_y = start_y + output_window
            # Y[:, ii, ff] = y[start_y:end_y, ff]

    return X#, Y
# X = windowed_dataset(np.array(data))
# X = X.reshape(-1,5,59)

# vrae = VRAE(sequence_length=5,
#             number_of_features = 59,
#             hidden_size = hidden_size,
#             hidden_layer_depth = hidden_layer_depth,
#             latent_length = latent_length,
#             batch_size = batch_size,
#             learning_rate = learning_rate,
#             n_epochs = n_epochs,
#             dropout_rate = dropout_rate,
#             optimizer = optimizer,
#             cuda = cuda,
#             print_every=print_every,
#             clip=clip,
#             max_grad_norm=max_grad_norm,
#             loss = loss,
#             block = block,
#             dload = dload)
#
# vrae.fit(X)

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

TIME_STEPS = 100

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(data)
print("Training input shape: ", x_train.shape)

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

