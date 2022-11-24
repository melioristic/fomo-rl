#!/bin/bash

#SBATCH --job-name=rl
#SBATCH --time=0-23:40:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=12

# output files
#SBATCH -o /work/anand/rl_1.out
#SBATCH -e /work/anand/rl_1.err

module purge
module load GCC/10.2.0

module load Anaconda3

source activate fomo_rl

# nnictl create -f --config brl_m_cfg.yml
srun python script_train_rl.py
