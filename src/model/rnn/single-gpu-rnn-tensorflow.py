import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

class RNNModel(tf.keras.Model):
    def __init__(self, step, units=50):
        super(RNNModel, self).__init__()
        self.step = step
        self.units = units
        self.rnn1 = SimpleRNN(units, activation='relu', return_sequences=True, input_shape=(1, step))
        self.rnn2 = SimpleRNN(units, activation='relu')
        self.dense = Dense(1)

    def call(self, x):
        x = self.rnn1(x)
        x = self.rnn2(x)
        return self.dense(x)

def load_and_preprocess_data():
    humidity = pd.read_csv("Data/humidity.csv")
    temp = pd.read_csv("Data/temperature.csv")
    pressure = pd.read_csv("Data/pressure.csv")

    # new york data
    humidity_NY = humidity[['datetime', 'New York']]
    temp_NY = temp[['datetime', 'New York']]
    pressure_NY = pressure[['datetime', 'New York']]

    # interpolate
    humidity_NY.interpolate(inplace=True)
    humidity_NY.dropna(inplace=True)
    temp_NY.interpolate(inplace=True)
    temp_NY.dropna(inplace=True)
    pressure_NY.interpolate(inplace=True)
    pressure_NY.dropna(inplace=True)

    merged_data = humidity_NY.merge(temp_NY, on='datetime').merge(pressure_NY, on='datetime')
    merged_data.columns = ['datetime', 'humidity', 'temperature', 'pressure']

    # test and train sets
    Tp = 7000
    train = np.array(temp_NY['New York'][:Tp])
    test = np.array(temp_NY['New York'][Tp:])

    train = train.reshape(-1, 1)
    test = test.reshape(-1, 1)

    # set step
    step = 8
    test = np.append(test, np.repeat(test[-1,], step))
    train = np.append(train, np.repeat(train[-1,], step))

    # rnn format
    trainX, trainY = convertToMatrix(train, step)
    testX, testY = convertToMatrix(test, step)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, trainY, testX, testY

def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)

def train(model, trainX, trainY, epochs, batch_size):
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

def test(model, testX, testY):
    test_loss = model.evaluate(testX, testY)
    print("Test Loss:", test_loss)
    return test_loss

def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def plot_prediction(predictions, testY):
    plt.figure(figsize=(10, 4))
    plt.plot(testY, label='True Value')
    plt.plot(predictions, label='Predicted Value')
    plt.title('Prediction')
    plt.ylabel('Temperature')
    plt.xlabel('Sample')
    plt.legend(loc='upper left')
    plt.show()

def main():
    # training settings
    parser = argparse.ArgumentParser(description='RNN for Time Series Prediction')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--step', type=int, default=8, help='time step for RNN')
    parser.add_argument('--units', type=int, default=50, help='number of units in RNN layers')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    # set up device
    if not args.no_cuda and tf.test.is_gpu_available(cuda_only=True):
        device_name = "GPU"
    elif not args.no_mps and tf.config.experimental.list_physical_devices('MPS'):
        device_name = "MPS"
    else:
        device_name = "CPU"
    tf.keras.backend.set_image_data_format('channels_last')
    print(f"Using {device_name} device")

    # load and preprocess data
    trainX, trainY, testX, testY = load_and_preprocess_data()

    # Define and compile model
    rnn_model = RNNModel(step=args.step, units=args.units)
    history = train(rnn_model, trainX, trainY, epochs=args.epochs, batch_size=args.batch_size)
    test_loss = test(rnn_model, testX, testY)

    # visualization
    plot_history(history)

    predictions = rnn_model.predict(testX)

    plot_prediction(predictions, testY)

    if args.save_model:
        rnn_model.save("rnn_model.h5")

if __name__ == "__main__":
    main()