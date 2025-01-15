#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0:10:00
#SBATCH --partition=bii
#SBATCH -A bii_dsc_community
#SBATCH --output=rp-%x-%j.out
#SBATCH --error=rp-%x-%j.err


ENV_PATH=/scratch/nww7sm/workdir/rp_dl
SCRIPT_PATH=/scratch/nww7sm/workdir/cylonplus/src/radical-dl/neural-forecast/graphing.py

module load gcc
module load python/.3.11.4
module load nvhpc/24.5
module load openmpi

source $ENV_PATH/bin/activate

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python $SCRIPT_PATH