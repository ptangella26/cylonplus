# Instructions 
# Running Cylon-DL on Rivanna

Arup Sarker (arupcsedu@gmail.com, djy8hg@virginia.edu)

## Install instructions

Rivanna is an HPC system offerbed by University of Virginia.
This will use custom dependencies of the system gcc, openmpi version.



```shell
ijob -c 2 -A bii_dsc_community -p bii-gpu --gres=gpu:v100:2 --time=0-10:00:00

git clone https://github.com/cylondata/cylon.git
cd cylon
```

## Data 

Data files for this tutorial have been taken from the article, ['Merge and Join DataFrames with
 Pandas in Python' by Shane Lynn](https://www.shanelynn.ie/merge-join-dataframes-python-pandas-index-1/) 
 that refers to  real data from the [KillBiller application](http://www.killbiller.com/).  

## C++ and Python Build

- Follow [Cylon docs](https://cylondata.org/docs/) for detailed building instructions, but in summary,  

```bash

conda create --solver=libmamba -p /scratch/djy8hg/env/gcylon_env -c pytorch -c nvidia -c rapidsai -c conda-forge  \
    python=3.11 rapids=24.06 \
    pytorch torchvision torchaudio pytorch-cuda=12.1 tensorflow dask-sql cmake \
    cython glog openmpi ucx gcc=12.2 gxx=12.2 gxx_linux-64=12.2 \
    setuptools pytest pytest-mpi mpi4py

pip install petastorm
pip install cloudmesh-common

conda activate /scratch/djy8hg/env/gcylon_env

./build.sh --conda_cpp --conda_python
```

- Run `demo_join.cpp` example 
```bash
./build/bin/demo_join
```

- For distributed execution using MPI 
```bash
mpirun -np <procs> ./build/bin/distributed_join
```

### Sequential Join

- Run `distributed_join.py` script
```bash
python src/cylon-dl/distributed_join.py
```

### Distributed Join

- For distributed execution using MPI 
```bash
mpirun -np <procs> <CYLON_HOME>/ENV/bin/python src/cylon-dl/distributed_join.py

mpirun -np 1 python src/cylon-dl/distributed_join.py
```

### Data Pre-Processing for Deep Learning with PyTorch

PyCylon pre-process the data starting from data loading and joining two tables
to formulate the features required for the data analytic carried out in PyTorch. 
PyCylon pre-process the data and releases the data as an Numpy NdArray at 
the end of the pipeline. 


 - Run distributed `cylon_pytorch_dl.py`
 
```bash
mpirun -n <procs> <CYLON_HOME>/ENV/bin/python cylonplus/src/cylon-dl/cylon_pytorch_dl.py
```

`Note: procs must be set such that, 0 < procs < 5`

- Install neuralforecast

```bash
conda install -c conda-forge neuralforecast --solver=libmamba
```