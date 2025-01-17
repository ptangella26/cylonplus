import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/us_carrier_passenger.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def graph_real_predict_error(model_path, model_name):
    def calc_error(x):
        if pd.notna(x[model_name]) and pd.notna(x['y'] != 'NaN'):
            return abs(x[model_name] - x['y'])
        else:
            return 0.0
        
    #pandas print config
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
         
    #load model and make predictions
    nf = NeuralForecast.load(model_path)
    y_df = load_data()

    y_hat_df = nf.predict(y_df.iloc[0:1])

    print(y_hat_df)
    y = 25

    while y < y_df.shape[0]:
        predicted_data = nf.predict(y_df.iloc[1:y])
        y_hat_df = pd.concat([y_hat_df, predicted_data], axis=0)
        y += 24

    y_df = y_df.reset_index(drop=True)

    #merge predictions dataframe and real data dataframe
    plot_df = pd.merge(y_df, y_hat_df, on=['ds'], how='outer').set_index('ds')
    plot_df['error'] = plot_df.apply(calc_error, axis=1)
    plot_df = plot_df.dropna()
    
    print(plot_df)
    widths = [d.days for d in np.diff(plot_df.index.tolist())]
    
    #graph real vs predicitons, error
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_df.index, plot_df['error'], color='red', label='Error', width=0.8*widths[0])
    ax.plot(plot_df.index, plot_df['y'], label='y (Real Values)', linewidth=2, color='blue')
    ax.plot(plot_df.index, plot_df[model_name], label='{} (Predictions)'.format(model_name), linewidth=2, color='orange')

    ax.set_title('AirPassengers Forecast ({})'.format(model_name), fontsize=10)
    ax.set_ylabel('Monthly Passengers', fontsize=10)
    ax.set_xlabel('Timestamp [t]', fontsize=10)
    ax.legend(loc='upper left', prop={'size': 10})

    plt.tight_layout()
    plt.savefig('/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/plots/{}.png'.format(model_name.lower()))
    plt.close()

if __name__ == '__main__':
    nbeats_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/nbeats/"
    graph_real_predict_error(nbeats_path, "NBEATS")

    deepar_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/deepar/"
    graph_real_predict_error(deepar_path, "DeepAR")

    nhits_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/nhits/"
    graph_real_predict_error(nhits_path, "NHITS")

    #itransformer_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/itransformer/"
    #graph_real_predict_error(itransformer_path, "iTransformer")

    gru_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/gru/"
    graph_real_predict_error(gru_path, "GRU")
    
    tcn_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/tcn/"
    graph_real_predict_error(tcn_path, "TCN")

    autoformer_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/autoformer/"
    graph_real_predict_error(autoformer_path, "Autoformer")

    fedformer_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/fedformer/"
    graph_real_predict_error(fedformer_path, "FEDformer")

    timesnet_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/timesnet/"
    graph_real_predict_error(timesnet_path, "TimesNet")

    vanillatransformer_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/vanilla_transformer/"
    graph_real_predict_error(vanillatransformer_path, "VanillaTransformer")

    patchtst_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/patchtst/"
    graph_real_predict_error(patchtst_path, "PatchTST")

    lstm_path = "/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/lstm/"
    graph_real_predict_error(lstm_path, "LSTM")

    




