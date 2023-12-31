# cylonplus
High-Performance Distributed Data frames for Machine Learning/Deep Learning Model


## Installation instructions
```
module load gcc-12.1.0  anaconda3

conda env create -f conda/gcylon.yml
conda activate gcylon_dev

export PATH=/u/djy8hg/anaconda3/envs/gcylon_dev/bin:$PATH LD_LIBRARY_PATH=/u/djy8hg/anaconda3/envs/gcylon_dev/lib:$LD_LIBRARY_PATH PYTHONPATH=/u/djy8hg/anaconda3/envs/gcylon_dev/lib/python3.8/site-packages

conda install pytorch torchvision torchtext torchaudio -c pytorch

cd src/model
python cnn.py

```
