import time
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.utils import AirPassengersDF
from neuralforecast.models import NHITS
from neuralforecast.models import iTransformer

def train_lstm():
    try:
        nf = NeuralForecast(
            models=[LSTM(h=12)],
            freq='M'
        )
        start_time = time.time()
        nf.fit(df=AirPassengersDF)
        nf.predict()
        end_time = time.time()

        task_time = end_time - start_time
        print(f"LSTM model training completed in {task_time:.2f} seconds")
        return str(task_time)
    except Exception as e:
        print(f"Error during LSTM model training: {str(e)}")
        return str(e)

def train_nhits():
    try:
        nf = NeuralForecast(
            models=[NHITS(input_size=24, h=12, max_steps=100, n_freq_downsample=[2, 1, 1])],
            freq='M'
        )
        start_time = time.time()
        nf.fit(df=AirPassengersDF)
        nf.predict()
        end_time = time.time()

        task_time = end_time - start_time
        print(f"NHITS model training completed in {task_time:.2f} seconds")
        return str(task_time)
    except Exception as e:
        print(f"Error during NHITS model training: {str(e)}")
        return str(e)

def train_transformer():
    try:
        nf = NeuralForecast(
            models=[iTransformer(input_size=24, h=12, n_series=1, max_steps=100)],
            freq='M'
        )
        start_time = time.time()
        nf.fit(df=AirPassengersDF)
        nf.predict()
        end_time = time.time()

        task_time = end_time - start_time
        print(f"Transformer model training completed in {task_time:.2f} seconds")
        return str(task_time)
    except Exception as e:
        print(f"Error during Transformer model training: {str(e)}")
        return str(e)




if __name__ == '__main__':
    train_lstm()
    #train_nhits()
    #train_transformer()