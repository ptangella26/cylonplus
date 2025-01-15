#!/usr/bin/env python3

import radical.pilot as rp


# ------------------------------------------------------------------------------
#
def task_state_cb(task, state):

  # print('  task %-30s: %s' % (task.uid, task.state))
    pass


# ------------------------------------------------------------------------------
#
@rp.pythontask
def hello_world(msg, sleep):
    import torch
    print(torch.cuda.is_available())  # Should return True if GPU is accessible
    print(torch.cuda.device_count())  # Number of GPUs available
    print(torch.cuda.current_device())  # ID of the current device
    print(torch.cuda.get_device_name(0))  # Name of the GPU (if any)

    import time
    print('hello %s: %.3f' % (msg, time.time()))
    time.sleep(sleep)
    print('hello %s: %.3f' % (msg, time.time()))
    return 'hello %s' % msg

# Define the NBEATS model training task
@rp.pythontask
def train_nbeats():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    from neuralforecast.utils import AirPassengersDF
    try:
        nf = NeuralForecast(
            models=[NBEATS(input_size=24, h=12, max_steps=100)],
            freq='M'
        )
        start_time = time.time()
        nf.fit(df=AirPassengersDF)
        nf.predict()
        end_time = time.time()

        task_time = end_time - start_time
        print(f"NBEATS model training completed in {task_time:.2f} seconds")
        return str(task_time)
    except Exception as e:
        print(f"Error during NBEATS model training: {str(e)}")
        return str(e)


# Define the DEEPAR model training task
@rp.pythontask
def train_deepar():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DeepAR
    from neuralforecast.utils import AirPassengersDF
    try:
        nf = NeuralForecast(
            models=[DeepAR(input_size=24, h=12, max_steps=100)],
            freq='M'
        )
        start_time = time.time()
        nf.fit(df=AirPassengersDF)
        nf.predict()
        end_time = time.time()

        task_time = end_time - start_time
        print(f"DEEPAR model training completed in {task_time:.2f} seconds")
        return str(task_time)
    except Exception as e:
        print(f"Error during DEEPAR model training: {str(e)}")
        return str(e)


# Define the LSTM model training task
@rp.pythontask

def train_lstm():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM
    from neuralforecast.utils import AirPassengersDF
    try:
        nf = NeuralForecast(
            models=[LSTM(h=24)],
            freq='M'
        )
        start_time = time.time()
        nf.fit(df=AirPassengersDF.iloc[0:120])
        nf.predict()
        end_time = time.time()

        task_time = end_time - start_time
        print(f"LSTM model training completed in {task_time:.2f} seconds")

        nf.save(path='/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/saved_models/lstm/',
                model_index=None, 
                overwrite=True,
                save_dataset=True)
        return str(task_time)
    except Exception as e:
        print(f"Error during LSTM model training: {str(e)}")
        return str(e)


# Define the lstm model training task
@rp.pythontask
def train_nhits():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.utils import AirPassengersDF
    from neuralforecast.models import NHITS

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




# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    session = rp.Session()
    try:
        pd_init = {'access_schema' : 'interactive',
                   'cores'         : 128,
                   'resource'      : 'uva.rivanna',
                   'runtime'       : 15}
        pd = rp.PilotDescription(pd_init)

        pmgr = rp.PilotManager(session=session)
        tmgr = rp.TaskManager(session=session)
        tmgr.register_callback(task_state_cb)

        pilot = pmgr.submit_pilots([pd])[0]
        tmgr.add_pilots(pilot)

        raptor = pilot.submit_raptors(rp.TaskDescription({'mode': rp.RAPTOR_MASTER}))[0]
        worker = raptor.submit_workers(
            [rp.TaskDescription({'mode': rp.RAPTOR_WORKER, 'ranks': 8}),
             rp.TaskDescription({'mode': rp.RAPTOR_WORKER, 'ranks': 8}),
             rp.TaskDescription({'mode': rp.RAPTOR_WORKER, 'ranks': 8})])


        td_func_2 = rp.TaskDescription({'mode': rp.TASK_FUNCTION,
                                        'function': train_lstm()})
        
        '''
        td_func_1 = rp.TaskDescription({'mode': rp.TASK_FUNCTION,
                                        'function': hello_world('client', 3)})

        td_func_2 = rp.TaskDescription({'mode': rp.TASK_FUNCTION,
                                        'function': 'hello',
                                        'args'    : ['raptor', 3]})
        '''
            
        #tasks = raptor.submit_tasks([td_func_1, td_func_2, td_func_3] * 256)
        tasks = raptor.submit_tasks([td_func_2])
        tmgr.wait_tasks([task.uid for task in tasks])

        for task in tasks:
            print('%s [%s]:\n%s \n%s' % (task.uid, task.state, task.stdout, task.stderr))

        raptor.rpc('stop')
        tmgr.wait_tasks(raptor.uid)
        print('%s [%s]: %s' % (raptor.uid, raptor.state, raptor.stdout))

    finally:
        session.close(download=False)

# ------------------------------------------------------------------------------