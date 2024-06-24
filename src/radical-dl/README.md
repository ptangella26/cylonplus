# Running Radical-Pilot on Rivanna

Arup Sarker (arupcsedu@gmail.com, djy8hg@virginia.edu)



## Install instructions for Radical Pilot

Rivanna is an HPC system offerbed by University of Virginia.
Use the same python environment "rp_dl" for radical-pilot, pytorch, cuda and other framework

```shell
module load anaconda
python -m venv $PWD/rp_dl
source $PWD/rp_dl/bin/activate
pip3 install radical.pilot
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip3 install cloudmesh-common

pip3 install petastorm
```
For checking all dependent library version:

```shell
radical-stack
```

Setup is done. Now let's execute scaling with cylon.

```shell
cd /some_path_to/cylonplus/src/radical-dl
```

## Modify the slurm scipt with your python virtual environment
```shell
source /path_to_your_venv/rp_dl/bin/activate

sbatch rp-dl.slurm
```

## Execute Slurm script 
```shell

sbatch rp-dl.slurm
```

If you want to make any change in the uva resource file(/some_path_to/radical.pilot/src/radical/pilot/configs) or any other places in the radical pilot source,

```shell
git clone https://github.com/radical-cybertools/radical.pilot.git
cd radical.pilot
```
For reflecting those change, you need to upgrade radical-pilot by,

```shell
pip install . --upgrade
```

To uninstall radical pilot, execute

```shell
pip uninstall radical.pilot
```