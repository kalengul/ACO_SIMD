#!/bin/sh
#SBATCH -p v100         # partition with GPUs
#SBATCH -n 1            # number of cores
#SBATCH --gres=gpu:1    # number of GPUs
#SBATCH -t 1-1:2:30     # days-hours:minutes:seconds
srun cuda_ACO
