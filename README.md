# lm-training

A skeleton for the training of transformer causal language models using HuggingFace libraries and models.  This library makes heavy use of [hydra](https://hydra.cc) for configuration management, so it is worth consulting that documentation as needed.  It also takes advantage of the HuggingFace ecosystem of libraries, so consulting the [transformer](https://huggingface.co/docs/transformers/index) and [tokenizer](https://huggingface.co/docs/tokenizers/index) package docs may also be beneficial.  Note that all language models supported by this package are ultimately [PyTorch modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).

# How to Run

This project demonstrates training (i) a simple whitespace tokenizer and (ii) two small LM's (a transformer and an RNN) on [wikipedia data from Gulordava et al 2018](https://github.com/facebookresearch/colorlessgreenRNNs/tree/main/data).  Their English data has been downloaded and saved in `data/wiki-en/`.  (Note: in principle, one can just use their supplied vocab file to initialize a whitespace tokenizer; we instead are training one just to illustrate the training of a tokenizer, which can be adapted for more complex types (BPE, etc).)

We will add more information on customizing configurations, building experiments, and things like that in the future.

## Tokenizer

To train the tokenizer: `python train_tokenizer.py`.  This will save a tokenizer to `models/tokenizer/word-level.json`.  The main configuration used for this script is `config/train-tokenizer.yaml`.

n.b.: This tokenizer is for demonstration purposes only.  Because the corpus contains special tokens e.g. \<unk\> and \<eos\> already, the trained tokenizer will parse these as '<', 'unk'/'eos', and '>'; also, it will fail to insert these tokens in new text.  This problem can be avoided by training the tokenizer on a corpus that does not already include special tokens.

## Causal Language Model

### Transformer

To train a causal LM with the OPT architecture: `python train_lm.py`.  The main configuration for this script is `config/train-lm.yaml`.  This will save outputs to a `checkpoints` sub-directory of hydra's default output directory (`outputs/DATE/TIME`), which is something that can be configured for your own experiments.

### RNN/LSTM/GRU

To train an LSTM, run:

```
python train_lm.py 'model=LSTM' 'model.vocab_size=30000' 'model.embedding_dim=???' 'model.hidden_dim=???' 'model.num_layers=???' 'model.dropout_p=???' '+model.tie_weights=???'
```

You can replace the parameters indicated by `???` with your desired values.  `model.vocab_size` should align with the vocab of the tokenizer.  The `embedding_dim`, `hidden_dim`, and `num_layers` are integers; `dropout_p` is a float between 0.0 and 1.0; and `tie_weights` is a boolean--- if `True`, `embedding_dim` should be equal to `hidden_dim`.


As with the transformer, this will save outputs to a `checkpoints` sub-directory of hydra's default output directory (`outputs/DATE/TIME`), which is something that can be configured for your own experiments.


To train a vanilla RNN or GRU, you can copy and modify the main LSTM config at `config/model/LSTM.yaml` so that the `rnn_type` argument says `RNN` or `GRU`, respectively, or you can override via the console, e.g. for a vanilla RNN:

```
python train_lm.py 'model=LSTM' 'model.rnn_type=RNN' 'model.vocab_size=30000' 'model.embedding_dim=???' 'model.hidden_dim=???' 'model.num_layers=???' 'model.dropout_p=???' '+model.tie_weights=???'
```

### Changing Config Defaults

You may consult the config hierarchy starting with `config/train_lm.yaml` / `config/train_tokenizer.yaml` to see what the default parameters are and how to override them.  As an example, while training an LM, you could change the tokenizer used and the data directory containing the train/validation/test data as follows:

```
python train_lm.py 'tokenizer.tokenizer_file=path/to/my_tokenizer/tokenizer.json' 'data.base_dir=path/to/my/data/' [...remaining arguments...]
```

For more information on how overriding config defaults works, consult the Hydra docs.

## Development & Contribution Guidelines

### Requirements

You will only need to make sure you have a recent version of Anaconda.  All other requirements are listed in `environment.yml`/`gpu_environment.yml` and installed/managed by conda.

### Patas Development Setup

On Patas, we have already created two environments for use with this project.  One is for use with GPU nodes, and the other with CPU nodes (including the head node that you would normally ssh into).

Instructions on setting up Conda on Patas can be found [here](https://www.shane.st/teaching/575/spr22/patas-gpu.pdf).  n.b.: you will have to go to the Anaconda website and find the link to the most recent version, as the link in this PDF is out of date.

#### Head Node Use (w/o Condor)

After installing conda as above, you may wish to test small changes while working on your own account on the head node.  To do so, you will want to first activate the CPU environment like so:

```sh
conda activate /projects/assigned/lm-inductive/envs/lm-training
```

As always, please abide by general Patas etiquete and avoid running jobs on the head node that require non-trivial amounts of CPU or memory usage.

#### Condor: CPU or GPU Nodes

There are two ways to tell Condor to use the environment when running a job.  The first works for CPU or GPU nodes, while the second works only for CPU nodes.

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

1. Create a fresh conda environment using `environment.yml`.  If you haven't done so for this project previously:
    ```sh
    conda env create -f environment.yml
    ```
    By default this will create a conda env whose name is indicated on the first line of the `environment.yml` file (presently, `lm-training`).  You can change this by adding the `-n` flag followed by the desired name of your environment.
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

If you need any new packages, install them with `conda install PACKAGE_NAME`.  Then, before committing, run:

```sh
conda env export --from-history | grep -vE "^(name|prefix):" > environment.yml
```

(Replace `environment.yml` with `gpu_environment.yml` as appropriate.)

This makes sure the `name:` and `prefix:` lines automatically created by Conda's `export` command are not included, since these values can vary by platform/machine.

Then make sure the updated `(gpu_)environment.yml` file is included with your commit.  Note: if you did not install the command with `conda install`, the above command will not work properly, due to the `--from-history` flag.  However, using this flag is necessary to ensure the `requirements.yml` file is platform-agnostic.  Therefore, please only install packages via `conda install` (or by manually adding requirements to the YAML files).

Optional, but recommended: before running `conda install` for new packages, run
```sh
conda config --set channel_priority strict
```

## Directory Structure

[WIP]
