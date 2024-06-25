#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH -A bii_dsc_community
#SBATCH --output=/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/radical-dl/logs/rp-dl-%x-%j.out
#SBATCH --error=/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/radical-dl/logs/rp-dl-%x-%j.err


module load anaconda


source /project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/rp_dl/bin/activate

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python model_training_with_rp.py
