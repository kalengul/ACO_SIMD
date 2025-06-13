#!/bin/sh
#SBATCH -p v100         # partition with GPUs
#SBATCH -n 4            # number of cores
#SBATCH --gres=gpu:4    # number of GPUs
#SBATCH -t 3-1:2:30     # days-hours:minutes:seconds
srun cuda_ACO
