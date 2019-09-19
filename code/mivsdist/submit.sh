#!/bin/bash 

# set number of nodes
#SBATCH -N 1
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=2000MB 
# time in format dd-hh:mm:ss
#SBATCH --time=02-00:00:00
# set nice descriptive name 
#SBATCH -J maxent
# queue name
#SBATCH -p bioth
# run as an array job (change number of tasks here)
#SBATCH --array=1-4
#SBATCH --output=test.out

# echo $SLURM_ARRAY_TASK_ID $HOSTNAME 
time python run.py $SLURM_ARRAY_TASK_ID
