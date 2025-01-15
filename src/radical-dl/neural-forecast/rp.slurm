#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --time=0:30:00
#SBATCH --partition=bii
#SBATCH -A bii_dsc_community
#SBATCH --output=rp-%x-%j.out
#SBATCH --error=rp-%x-%j.err
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:a100:1


ENV_PATH=/scratch/nww7sm/workdir/rp_dl
SCRIPT_PATH=/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/raptor_simple_.py
#SCRIPT_PATH=/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/radical-dl/hydrology/neural-forecast/raptor-neural-forecast.py

module load gcc
module load python/.3.11.4
module load nvhpc/24.5
module load openmpi

source $ENV_PATH/bin/activate

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python $SCRIPT_PATH
