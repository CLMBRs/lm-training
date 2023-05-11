#!/bin/bash

# Base script for updating or creating the conda environments for use with this project.
# Usage: env_install+update.sh [-p </environment/prefix/path> | -n <environment_name>] (<path/to/environment.yml>)

# -e: immediately exit if any program errors while running the script
# -u: prohibit undefined variables
# -x: print each command being executed
# -o pipefail: if any program in a pipeline errors, its error code is the error code of the whole script
# set -euxo pipefail # for debugging
set -eo pipefail

conda_env_exists_prefix(){
    conda env list | grep -P "^\S*\s+${@}$" >/dev/null 2>/dev/null
}

conda_env_exists_name(){
    conda env list | grep -P "^${@}\s" >/dev/null 2>/dev/null
}

if [[ $# -gt 3 || $# -lt 2 ]] || [[ $1 != "-p" && $1 != "-n" ]]
then
    echo "Usage: env_install+update.sh [-p </environment/prefix/path> | -n <environment_name>] (<path/to/environment.yml>)"
    exit 1
elif [[ $1 = "-p" ]]
then
    ENV_CHECK="conda_env_exists_prefix"
    ENV_ID=$(realpath "$2")
elif [[ $1 = "-n" ]]
then
    ENV_CHECK="conda_env_exists_name"
    ENV_ID="$2"
fi

ENV_ID_ARGS=${@:1:2}
YML_FILE=$(realpath "${3:-../environment.yml}")

echo "Using environment $(echo $ENV_CHECK | grep -Po "(name|prefix)"): $ENV_ID"
echo "Using YML file: $YML_FILE"

# if $CONDA_EXE is not defined for some reason because of an unusual (mini)conda installation, change the following line:
# CONDA_PROFILE=path/to/anaconda3/etc/profile.d/conda.sh
CONDA_PROFILE=$(realpath $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh)

if [[ ! -f $CONDA_PROFILE ]] 
then
    echo "Conda profile not found at $CONDA_PROFILE. Exiting."
    exit 1
fi

if [[ ! -f $YML_FILE ]] 
then
    echo "Environment file not found at $YML_FILE. Exiting."
    exit 1
fi

echo "Sourcing anaconda profile from $CONDA_PROFILE"
source $CONDA_PROFILE
echo "Sourced Conda.sh script. Now activating base environment"

conda activate base
if [[ $CONDA_DEFAULT_ENV && "base" = $CONDA_DEFAULT_ENV ]]
then
    echo "Successfully activated base environment."
else
    echo "Could not activate base environment. Aborting."
    exit 1
fi

if ! $ENV_CHECK $ENV_ID
then
    echo "Conda environment does not exist. Creating environment at $ENV_ID."
    echo "Creating with command conda create $ENV_ID_ARGS --strict-channel-priority --yes"
    conda create $ENV_ID_ARGS --strict-channel-priority --yes
else
    echo "Conda environment exists. Proceeding with update."
fi

if ! $ENV_CHECK $ENV_ID # check it was created successfully above
then
    echo "Failed to create Conda environment. Aborting."
    exit 1
fi

echo "Now activating Conda environment: $ENV_ID..."

# deactivate base environment
conda deactivate

conda activate $ENV_ID

if [[ $CONDA_DEFAULT_ENV && $ENV_ID = $CONDA_DEFAULT_ENV ]] || [[ $CONDA_PREFIX && $ENV_ID = $CONDA_PREFIX ]]
then
    echo "Successfully activated environment at $ENV_ID. Conda env is now $CONDA_DEFAULT_ENV"
else
    echo "Could not activate environment at $ENV_ID; instead activated: $CONDA_DEFAULT_ENV. Aborting."
    exit 1
fi

if [[ ! $(conda config --env --show channel_priority | grep "strict") ]]
then
    echo "Channel priority not strict; aborting."
    exit 1
fi

# temporary
conda config --env --set always_yes true

echo "Updating environment to use specifications in $YML_FILE"
conda env update -f $YML_FILE $ENV_ID_ARGS --prune --json

# reset
conda config --env --remove-key always_yes

echo "Done updating environment. You may want to double check the desired packages are available."
conda deactivate
