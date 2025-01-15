#!/usr/bin/env python3

import os
import sys
import radical.pilot as rp
import radical.utils as ru
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF

# Set up the current working directory and config directory
PWD = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.abspath('%s/../' % PWD)

# ------------------------------------------------------------------------------
# Task state callback to print task status updates
def task_state_cb(task, state):
    print('  task %-30s: %s' % (task.uid, task.state))

# ------------------------------------------------------------------------------
# Define the model training task
@rp.pythontask
def train_model(model_name, input_size, h, max_steps):
    model_class = globals()[model_name]
    model = model_class(input_size=input_size, h=h, max_steps=max_steps)
    nf = NeuralForecast(models=[model], freq='M')

    start_time = time.time()
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end_time = time.time()

    task_time = end_time - start_time
    print(f"Task {model_name} completed in {task_time:.2f} seconds")
    return Y_hat_df, task_time

# ------------------------------------------------------------------------------
# Main execution block
if __name__ == '__main__':
    report = ru.Reporter(name='radical.pilot')
    report.title('Time Series Forecasting with RAPTOR (RP version %s)' % rp.version)

    # Determine the resource to use, defaulting to 'uva.rivanna' if not provided
    if   len(sys.argv)  > 2: report.exit('Usage:\t%s [resource]\n\n' % sys.argv[0])
    elif len(sys.argv) == 2: resource = sys.argv[1]
    else                   : resource = 'uva.rivanna'

    session = rp.Session()
    try:
        # Load the configuration file
        config = ru.read_json('%s/config.json' % CONFIG_DIR)

        # Initialize the PilotManager with the session
        pmgr = rp.PilotManager(session=session)

        # Initialize pilot description using the configuration and resource specified
        pd_init = {
            'resource'      : resource,
            'runtime'       : 60,  # pilot runtime (min)
            'exit_on_error' : True,
            'project'       : config[resource].get('project', None),
            'queue'         : config[resource].get('queue',   None),
            'access_schema' : config[resource].get('schema',  None),
            'cores'         : config[resource].get('cores',   1),
            'gpus'          : config[resource].get('gpus', 0)  # Add GPU support if available
        }
        report.info(pd_init)
        pdesc = rp.PilotDescription(pd_init)

        # Submit the pilot
        pilot = pmgr.submit_pilots(pdesc)

        # Initialize TaskManager and associate it with the pilot
        tmgr = rp.TaskManager(session=session)
        tmgr.add_pilots(pilot)

        # Register the task state callback
        tmgr.register_callback(task_state_cb)

        # Set up RAPTOR master and workers
        raptor = pilot.submit_raptors(rp.TaskDescription({'mode': rp.RAPTOR_MASTER}))[0]
        workers = raptor.submit_workers(rp.TaskDescription({'mode': rp.RAPTOR_WORKER}))

        # Prepare dataset
        Y_df = AirPassengersDF
        Y_train_df = Y_df[Y_df.ds <= '1959-12-31']
        Y_test_df = Y_df[Y_df.ds > '1959-12-31']
        horizon = len(Y_test_df)

        # Define tasks for the models
        td_func_1 = rp.TaskDescription({
            'mode': rp.TASK_FUNCTION,
            'function': train_model('NBEATS', 2 * horizon, horizon, 50)
        })

        td_func_2 = rp.TaskDescription({
            'mode': rp.TASK_FUNCTION,
            'function': train_model('NHITS', 2 * horizon, horizon, 50)
        })

        # Submit tasks to RAPTOR
        tasks = raptor.submit_tasks([td_func_1, td_func_2])

        # Wait for tasks to complete
        tmgr.wait_tasks([task.uid for task in tasks])

        # Print task output
        for task in tasks:
            print('%s [%s]:\n%s' % (task.uid, task.state, task.stdout))

        # Stop the RAPTOR master and wait for it to complete
        raptor.rpc('stop')
        tmgr.wait_tasks(raptor.uid)
        print('%s [%s]: %s' % (raptor.uid, raptor.state, raptor.stdout))

    finally:
        # Clean up the session
        session.close(download=False)

    # Post-process and plot the results
    Y_hat_df = pd.concat([task.return_value[0] for task in tasks]).reset_index(drop=True)
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
    plot_df[['y', 'NBEATS', 'NHITS']].plot(ax=ax, linewidth=2)
    ax.set_title('AirPassengers Forecast', fontsize=22)
    ax.set_ylabel('Monthly Passengers', fontsize=20)
    ax.set_xlabel('Timestamp [t]', fontsize=20)
    ax.legend(prop={'size': 15})
    ax.grid()

    # Save the plot
    plt.savefig('air_passengers_forecast.png')
    print("Forecast plot saved as air_passengers_forecast.png")

    # Print some statistics
    for model in ['NBEATS', 'NHITS']:
        mse = np.mean((Y_test_df['y'] - Y_hat_df[model]) ** 2)
        print(f"{model} Mean Squared Error: {mse:.2f}")
