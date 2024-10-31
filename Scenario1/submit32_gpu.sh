#!/bin/bash
#SBATCH --account=def-wperrie
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=32G          # Request the full memory of the node
#SBATCH --time=2:00:00

module purge
module load python/3.10 scipy-stack
module load cuda cudnn 

##virtualenv --no-download ~/mypython/env
source ~/mypython/env/bin/activate
###virtualenv --no-download tensorflow
source tensorflow/bin/activate
###pip install --no-index tensorflow==2.8
pip install --no-index keras
pip install --no-index tqdm
pip install --no-index xarray
pip install  --no-index numpy 

###pip install h5io h5py h5netcdf --no-index
python train.py


salloc --account=def-wperrie --gres=gpu:4 --cpus-per-task=6 --mem=64G --time=0-06:00
