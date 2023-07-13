#!/bin/bash

#SBATCH --job-name=lm-training_install

#SBATCH --account=stf
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=slurm_output/%x_%j.out # where STDOUT goes
#SBATCH --error=slurm_output/%x_%j.err # where STDERR goes

# Make sure you submit this script from within the env_management directory; otherwise, utilize the SBATCH --chdir
# command or CLI flag appropriately.

./env_install+update.sh -n lm-training ../environment.yml
