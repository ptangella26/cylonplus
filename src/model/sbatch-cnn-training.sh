
#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:60:00
#SBATCH --job-name="CNN Training" 
#SBATCH --partition=main
#SBATCH --error="%j-stderr.txt"
#SBATCH --output="%j-stdout.txt"
#
#source ~/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

conda activate cyp-venv
module load java8-1.8.0_202  

#nodes= $( scontrol show hostnames $SLURM_JOB_NODELIST )
#nodes_array=($nodes)
#head_node=${nodes_array[0]}
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

#echo Node IP: $head_node_ip
#export LOGLEVEL=INFO

torchrun --nnodes 4 --nproc_per_node 1 parquet-mg-cnn-tr.py --total_epochs 3 --save_every 1 --batch_size 32
#--rdzv_id $RANDOM \
#--rdzv_backend c10d \
#--rdzv_endpoint $head_node_ip:29500 \
