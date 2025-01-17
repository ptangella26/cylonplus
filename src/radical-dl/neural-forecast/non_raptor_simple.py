#!/usr/bin/env python3
from neuralforecast import NeuralForecast

def train_nbeats():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import logging
    import pandas as pd
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[NBEATS(input_size=24, h=24, max_steps=100)],
            freq='M'
        )
        df = df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['NBEATS'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"NBEATS model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
       
        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/nbeats/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        print("saved model")
        return str(task_time)
    except Exception as e:
        print(f"Error during NBEATS model training: {str(e)}")
        return str(e)


def train_deepar():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DeepAR
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[DeepAR(
                input_size=24, 
                h=24,
                max_steps=100,
                loss=DistributionLoss(distribution='Normal', level=[80, 90], return_params=False),
                learning_rate=0.005
            )],
            freq='M'
        )
        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['DeepAR'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"DeepAR model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/deepar/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)

        return str(task_time)
    except Exception as e:
        print(f"Error during DeepAR model training: {str(e)}")
        return str(e)

def train_lstm():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[LSTM(h=24)],
            freq='M'
        )
        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['LSTM'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"LSTM model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/lstm/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during LSTM model training: {str(e)}")
        return str(e)

def train_nhits():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.utils import AirPassengersDF
    from neuralforecast.models import NHITS
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[NHITS(input_size=24, h=24, max_steps=100, n_freq_downsample=[2, 1, 1])],
            freq='M'
        )

        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])

        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['NHITS'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"NHITS model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/nhits/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)

        return str(task_time)
    except Exception as e:
        print(f"Error during NHITS model training: {str(e)}")
        return str(e)

def train_itransformer():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import iTransformer
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[iTransformer(input_size=24, h=24, n_series=1, max_steps=100)],
            freq='M'
        )

        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])

        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['iTransformer'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"iTransformer model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/itransformer/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during iTransformer model training: {str(e)}")
        return str(e)

def train_gru():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import GRU
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[GRU(h=24, max_steps=100)],
            freq='M'
        )

        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])

        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['GRU'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"GRU model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/gru/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during GRU model training: {str(e)}")
        return str(e)


def train_tcn():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TCN
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[TCN(h=24, max_steps=100)],
            freq='M'
        )
        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['TCN'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"TCN model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/tcn/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during TCN model training: {str(e)}")
        return str(e)

def train_autoformer():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import Autoformer
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[Autoformer(input_size=24, h=24, max_steps=100)],
            freq='M'
        )
        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['Autoformer'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"Autoformer model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/autoformer/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during Autoformer model training: {str(e)}")
        return str(e)

def train_fedformer():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import FEDformer
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[FEDformer(input_size=24, h=24, max_steps=100)],
            freq='M'
        )

        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])

        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['FEDformer'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"FEDformer model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/fedformer',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during FEDformer model training: {str(e)}")
        return str(e)

def train_timesnet():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TimesNet
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[TimesNet(input_size=24, h=24, max_steps=100)],
            freq='M'
        )

        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])

        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['TimesNet'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"TimesNet model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/timesnet/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during TimesNet model training: {str(e)}")
        return str(e)

def train_vanillatransformer():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import VanillaTransformer
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[VanillaTransformer(input_size=24, h=24, max_steps=100)],
            freq='M'
        )

        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])

        start_time = time.time()
        nf.fit(df=df.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['VanillaTransformer'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"VanillaTransformer model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/vanilla_transformer/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during VanillaTransformer model training: {str(e)}")
        return str(e)

def train_patchtst():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    from neuralforecast.utils import AirPassengersDF
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        nf = NeuralForecast(
            models=[PatchTST(input_size=24, h=24, max_steps=100)],
            freq='M'
        )

        df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
        df['ds'] = pd.to_datetime(df['ds'])

        start_time = time.time()
        nf.fit(df=AirPassengersDF.iloc[0:274])
        predictions = nf.predict()
        end_time = time.time()

        y_true = df['y'].values[-24:]
        y_pred = predictions['PatchTST'].values[-24:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

        task_time = end_time - start_time
        print(f"PatchTST model training completed in {task_time:.2f} seconds")
        print(f"Evaluation Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/patchtst/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        
        return str(task_time)
    except Exception as e:
        print(f"Error during PatchTST model training: {str(e)}")
        return str(e)

# ------------------------------------------------------------------------------
#
if __name__ == '__main__':
    train_nbeats()
    train_deepar()
    train_lstm()
    train_nhits()
    train_itransformer()
    train_gru()
    train_tcn()
    train_autoformer()
    train_fedformer()
    train_timesnet()
    train_vanillatransformer()
    train_patchtst()