#!/usr/bin/env bash
# slurm template for serial jobs
#SBATCH --job-name=finalproject # Job name
#SBATCH --output=finalproject-%j.out
#SBATCH --mail-user=jgilyard@middlebury.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=4g
#SBATCH --partition=gpu-standard
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 

# Your calculations here 
python train.py --weights bert-base-multilingual-cased --filename trained_bert_weights.pth
# End of job info 
echo "Ending: "`date +"%D %T"`