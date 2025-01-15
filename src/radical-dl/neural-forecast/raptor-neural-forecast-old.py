import os
import radical.pilot as rp
import radical.utils as ru
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF
import time  # Import the time module

# Disable animated output
os.environ['RADICAL_REPORT_ANIME'] = 'False'

# Set up the session and managers
session = rp.Session()
pmgr = rp.PilotManager(session)
tmgr = rp.TaskManager(session)

# Submit a pilot
pilot_description = rp.PilotDescription({
    'resource': 'uva.rivanna',
    'runtime': 60,
    'cores': 17,
    'exit_on_error': False
})
pilot = pmgr.submit_pilots(pilot_description)

# Add the pilot to the task manager and wait for it to become active
tmgr.add_pilots(pilot)
pilot.wait(rp.PMGR_ACTIVE)
print('Pilot is up and running')

# Set up RAPTOR master and workers
master_descr = {'mode': rp.RAPTOR_MASTER, 'named_env': 'rp'}
worker_descr = {'mode': rp.RAPTOR_WORKER, 'named_env': 'rp'}

raptor = pilot.submit_raptors([rp.TaskDescription(master_descr)])[0]
workers = raptor.submit_workers([rp.TaskDescription(worker_descr),
                                 rp.TaskDescription(worker_descr)])

# Split data and declare panel dataset
Y_df = AirPassengersDF
Y_train_df = Y_df[Y_df.ds <= '1959-12-31']  # 132 train
Y_test_df = Y_df[Y_df.ds > '1959-12-31']  # 12 test

horizon = len(Y_test_df)

@rp.pythontask
def train_model(model_name, input_size, h, max_steps):
    import time  # Import the time module inside the task
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, NHITS
    
    model_class = globals()[model_name]
    model = model_class(input_size=input_size, h=h, max_steps=max_steps)
    nf = NeuralForecast(models=[model], freq='M')
    
    start_time = time.time()  # Start timing the task
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end_time = time.time()  # End timing the task
    
    task_time = end_time - start_time
    print(f"Task {model_name} completed in {task_time:.2f} seconds")
    return Y_hat_df, task_time

# Track overall time
overall_start_time = time.time()

# Create tasks for each model
tasks = []
models = [
    ('NBEATS', 2 * horizon, horizon, 50),
    ('NBEATS', 2 * horizon, horizon, 50),
    ('NBEATS', 2 * horizon, horizon, 50),
    ('NBEATS', 2 * horizon, horizon, 50),
    ('NBEATS', 2 * horizon, horizon, 50),

    ('NHITS', 2 * horizon, horizon, 50),
    ('NHITS', 2 * horizon, horizon, 50),
    ('NHITS', 2 * horizon, horizon, 50),
    ('NHITS', 2 * horizon, horizon, 50),
    ('NHITS', 2 * horizon, horizon, 50),
]

for model_params in models:
    td = rp.TaskDescription({
        'mode': rp.TASK_FUNCTION,
        'function': train_model(*model_params)
    })
    tasks.append(td)

print("Number of tasks:", len(tasks))
# Submit tasks to RAPTOR
submitted_tasks = raptor.submit_tasks(tasks)

# Wait for tasks to complete
tmgr.wait_tasks([task.uid for task in submitted_tasks])

# Collect results
Y_hat_dfs = []
task_times = []
for task in submitted_tasks:
    print(task)
    Y_hat_df, task_time = task.return_value
    Y_hat_dfs.append(Y_hat_df)
    task_times.append(task_time)

# Print individual task times
for i, task_time in enumerate(task_times):
    print(f"Task {i+1} time: {task_time:.2f} seconds")

# Track overall time
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"Overall execution time: {overall_time:.2f} seconds")

# Close the session
session.close()
