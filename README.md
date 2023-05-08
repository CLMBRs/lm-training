# lm-training

- `conda env create -f environment.yml` for local development; if on a GPU machine, use `gpu_environment.yml` instead
- Train a tokenizer: `python src/train_tokenizer.py data.base_dir=XXX` (override base dir and other configs as needed)