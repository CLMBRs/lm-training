# lm-training

A skeleton for the training of transformer causal language models using HuggingFace libraries and models.

# How to Run

This project demonstrates training (i) a simple whitespace tokenizer and (ii) a small transformer LM on [wikipedia data from Gulordava et al 2018](https://github.com/facebookresearch/colorlessgreenRNNs/tree/main/data).  Their English data has been downloaded and saved in `data/wiki-en/`.  (Note: in principle, one can just use their supplied vocab file to initialize a whitespace tokenizer; we instead are training one just to illustrate the training of a tokenizer, which can be adapted for more complex types (BPE, etc).)

A simple Trainer with a BERT model, with the Rotten Tomatoes HF dataset for its training/validation dataset, can be loaded with the following command:

```
python src/train_lm.py '+trainer.model.config.pretrained_model_name_or_path=gpt2' '+dataset.path=rotten_tomatoes'
```

## Development & Contribution Guidelines

### Requirements

You will only need to make sure you have a recent version of Anaconda. All other requirements are listed in `environment.yml`/`gpu_environment.yml` and installed/managed by conda.

### Patas Development Setup

On Patas, we have already created two environments for use with this project. One is for use with GPU nodes, and the other with CPU nodes (including the head node that you would normally ssh into).

Instructions on setting up Conda on Patas can be found [here](https://www.shane.st/teaching/575/spr22/patas-gpu.pdf). n.b.: you will have to go to the Anaconda website and find the link to the most recent version, as the link in this PDF is out of date.

#### Head Node Use (w/o Condor)

After installing conda as above, you may wish to test small changes while working on your own account on the head node. To do so, you will want to first activate the CPU environment like so:

```sh
conda activate /projects/assigned/lm-inductive/envs/lm-training
```

As always, please abide by general Patas etiquete and avoid running jobs on the head node that require non-trivial amounts of CPU or memory usage.

#### Condor: CPU or GPU Nodes

There are two ways to tell Condor to use the environment when running a job. The first works for CPU or GPU nodes, while the second works only for CPU nodes.

##### Method A

1. In your Condor submit file, add a line saying `getenv = False` (or edit if `getenv` is already there)
1. Add these two lines near/at the top of the shell script (executable) that you are submitting to Condor, adjusting the first line if your condor installation is elsewhere:

For CPU nodes:
```sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/lm-inductive/envs/lm-training
```

For GPU nodes:
```sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/lm-inductive/envs/gpu-lm-training
```

Note that you will also have to edit your Condor submit file to request GPU nodes; for instructions regarding how to do that, see the document linked to near the top of this README that also contain the instructions for installing conda on Patas.

##### Method B
n.b.: This only works for CPU nodes.

1. While logged into your Patas account on the Patas node, run `conda activate /projects/assigned/lm-inductive/envs/lm-training` (unless you are already working within this environment)
1. Add `getenv = True` to your Condor submit file
1. Call `condor_submit` with the submit file as per usual.

### Hyak Development Setup

TODO

### Local Development Setup

1. Create a fresh conda environment using `environment.yml`. If you haven't done so for this project previously:
    ```sh
    conda env create -f environment.yml
    ```
    By default this will create a conda env whose name is indicated on the first line of the `environment.yml` file (presently, `lm-training`). You can change this by adding the `-n` flag followed by the desired name of your environment.
1. After the environment is created, whenever you want to work on this project, first activate the environment:
    ```sh
    conda activate lm-training
    ```
1. When you are done, you can exit the environment with `conda deactivate`.
1. If you pull code from the repo and the `environment.yml` file has changed, update your environment by running the following (after activating the environment):
    ```sh
    conda env update -f environment.yml --prune
    ```

### Contribution Guidelines

For any non-trivial changes, please work on your own branch rather than on `main` and submit a PR when you are ready to merge your changes.

If you need any new packages, install them with `conda install PACKAGE_NAME`. Then, before committing, run:

```sh
conda env export --from-history | grep -vE "^(name|prefix):" > environment.yml
```

(Replace `environment.yml` with `gpu_environment.yml` as appropriate.)

This makes sure the `name:` and `prefix:` lines automatically created by Conda's `export` command are not included, since these values can vary by platform/machine.

Then make sure the updated `(gpu_)environment.yml` file is included with your commit. Note: if you did not install the command with `conda install`, the above command will not work properly, due to the `--from-history` flag. However, using this flag is necessary to ensure the `requirements.yml` file is platform-agnostic. Therefore, please only install packages via `conda install` (or by manually adding requirements to the YAML files).

Optional, but recommended: before running `conda install` for new packages, run
```sh
conda config --set channel_priority strict
```

## Directory Structure

[WIP]
