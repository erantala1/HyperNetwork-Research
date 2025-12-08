#!/bin/bash  
### Job Name
#PBS -N Hypernetwork_FNO
#PBS -e Hypernetwork_FNO.err
### Charging account
#PBS -A UCSC0009
#PBS -l walltime=06:00:00
#PBS -q main
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:gpu_type=a100:mem=100GB

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /glade/work/erantala/conda-envs/jacobian_env

cd /glade/derecho/scratch/erantala/project_runs/code

python -u nn_train_multistep.py