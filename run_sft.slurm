#!/bin/bash
#SBATCH --job-name=neox # create a short name for your job
#SBATCH -p batch
#SBATCH --reservation={your qos}
#SBATCH --qos {your qos}
#SBATCH -N 4 # node count
#SBATCH --ntasks-per-node 8 # number of tasks to run per node
#SBATCH --cpus-per-task 10 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
###SBATCH --gres=gpu:8
#SBATCH --gpus-per-node 8 # total cpus for job
##SBATCH -o logs/%x-%j.log # output and error log file names (%x for job id)
##SBATCH -e logs/%x-%j.err # output and error log file names (%x for job id)


echo $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

scontrol show hostname $SLURM_JOB_NODELIST | sed 's/$/ slots=8/' > hostfile

NCCL_DEBUG=INFO
GPUS_PER_NODE=8
NNODES=4
export SLURM_NTASKS=$(($GPUS_PER_NODE*$NNODES))

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000

python ./deepy.py ./train.py ./custom_configs/llama2/70B_reset_mask_global.yml #{your config}