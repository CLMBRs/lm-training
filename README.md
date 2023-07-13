# lm-training

A skeleton for the training of transformer _and recurrent_ causal language models using HuggingFace libraries and models.  This library makes heavy use of [hydra](https://hydra.cc) for configuration management, so it is worth consulting that documentation as needed.  It also takes advantage of the HuggingFace ecosystem of libraries, so consulting the [transformer](https://huggingface.co/docs/transformers/index) and [tokenizer](https://huggingface.co/docs/tokenizers/index) package docs may also be beneficial.  Note that all language models supported by this package are ultimately [PyTorch modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).

# How to Run

This project demonstrates training (i) a simple whitespace tokenizer and (ii) two small LM's (a transformer and an RNN) on [wikipedia data from Gulordava et al 2018](https://github.com/facebookresearch/colorlessgreenRNNs/tree/main/data).  Their English data can be downloaded and saved using the download script at `data/wiki/download.sh`.  (Note: in principle, one can just use their supplied vocab file to initialize a whitespace tokenizer; we instead are training one just to illustrate the training of a tokenizer, which can be adapted for more complex types (BPE, etc).).  The following examples require that the data already be downloaded; if using other data, consult the `data` struct in `config/train-lm.yaml` and the `data/raw.yaml` config file and override the appropriate values.

We will add more information on customizing configurations, building experiments, and things like that in the future.

## Tokenizer

To train the tokenizer: `python train_tokenizer.py`.  This will save a tokenizer to `models/tokenizer/word-level.json`.  The main configuration used for this script is `config/train-tokenizer.yaml`.

n.b.: This tokenizer is for demonstration purposes only.  Because the corpus contains special tokens e.g. \<unk\> and \<eos\> already, the trained tokenizer will parse these as '<', 'unk'/'eos', and '>'; also, it will fail to insert these tokens in new text.  This problem can be avoided by training the tokenizer on a corpus that does not already include special tokens.

## Causal Language Model

### Transformer

To train a causal LM with the OPT architecture:
```sh
python train_lm.py --config-name train-causal-transformer
```

The main configuration for this script is `config/train-lm.yaml`.  This will save outputs to a `checkpoints` sub-directory of hydra's default output directory (`outputs/DATE/TIME`), which is something that can be configured for your own experiments.

### RNN/LSTM/GRU

To train an LSTM, run:

```sh
python train_lm.py --config-name train-lstm
```

To change the hyperparameters of the network, consult the arguments in `model/LSTM.yaml` and override them accordingly.  Some notes:
  * `model.vocab_size` should align with the vocab of the tokenizer.
  * The `embedding_dim`, `hidden_dim`, and `num_layers` are integers; `dropout_p` is a float between 0.0 and 1.0; and `tie_weights` is a boolean.
  * If `tie_weights` is `True`:`embedding_dim` should be equal to `hidden_dim` for a unidirectional RNN and double it for a bidirectional RNN.


As with the transformer, this will save outputs to a `checkpoints` sub-directory of hydra's default output directory (`outputs/DATE/TIME`), which is something that can be configured for your own experiments.

To train a vanilla RNN or GRU, you can override the `rnn_type` argument, e.g. for a vanilla RNN:

```sh
python train_lm.py --config-name train-lstm 'model.rnn_type=RNN'
```

### Changing Config Defaults

You may consult the config hierarchy starting with `config/train_lm.yaml` / `config/train_tokenizer.yaml` to see what the default parameters are and how to override them.  As an example, while training an LM, you could change the tokenizer used and the data directory containing the train/validation/test data as follows:

```sh
python train_lm.py 'tokenizer.tokenizer_file=path/to/my_tokenizer/tokenizer.json' 'data.base_dir=path/to/my/data/' [...remaining arguments...]
```

For more information on how overriding config defaults works, consult the Hydra docs.

## Development & Contribution Guidelines

### Requirements

You will only need to make sure you have a recent version of Anaconda.  All other requirements are listed in `environment.yml`/`gpu_environment.yml` and installed/managed by conda.

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
