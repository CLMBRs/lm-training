#!/bin/bash

#SBATCH --job-name=gpu-lm-training_install

#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --time=3:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=slurm_output/%x_%j.out # where STDOUT goes
#SBATCH --error=slurm_output/%x_%j.err # where STDERR goes

# Make sure you submit this script from within the env_management directory; otherwise, utilize the SBATCH --chdir
# command or CLI flag appropriately.

./env_install+update.sh -n gpu-lm-training ../gpu_environment.yml
