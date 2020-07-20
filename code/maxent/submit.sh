#!/bin/bash 

# set number of nodes
#SBATCH -N 1
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=8000MB 
#SBATCH --time=50
# set nice descriptive name 
#SBATCH -J maxent
# queue name
#SBATCH -p bioth
# run as an array job (change number of tasks here)
#SBATCH --array=1-32
#SBATCH --output=test.out

echo $SLURM_ARRAY_TASK_ID $HOSTNAME 
time python fit_all.py $SLURM_ARRAY_TASK_ID
