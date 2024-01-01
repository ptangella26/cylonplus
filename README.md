# cylonplus
High-Performance Distributed Data frames for Machine Learning/Deep Learning Model


## Installation instructions
```
module load anaconda3

conda create -n cyp-venv python=3.11
conda activate cyp-venv

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

cd src/model
python cnn.py

```
