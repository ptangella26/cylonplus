# cylonplus
High-Performance Distributed Data frames for Machine Learning/Deep Learning Model


## Installation instructions
```
ssh your_computing_id@gpusrv08 -J your_computing_id@portal.cs.virginia.edu
git clone https://github.com/arupcsedu/cylonplus.git
cd cylonplus
module load anaconda3

conda create -n cyp-venv python=3.11
conda activate cyp-venv

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
DIR=/u/djy8hg/anaconda3/envs/cyp-venv 

export CUDA_HOME=$DIR/bin
export PATH=$DIR/bin:$PATH LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH PYTHONPATH=$DIR/lib/python3.11/site-packages 

cd src/model
python multi-gpu-cnn.py

```
