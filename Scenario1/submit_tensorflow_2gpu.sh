#!/bin/bash
#SBATCH --account=def-wperrie
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --mem=8G      
#SBATCH --time=0-00:30
#SBATCH --output=%N-%j.out

module load python/3
virtualenv --no-download $SLURM_TMPDIR/env
source ~/mypython/env/bin/activate
pip install --no-index tensorflow

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

python tensorflow-singleworker.py
