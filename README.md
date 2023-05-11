# lm-training

- `conda env create -f environment.yml` for local development; if on a GPU machine, use `gpu_environment.yml` instead
- `pip install hydra`
- Train a tokenizer: `python src/train_tokenizer.py data.base_dir=XXX` (override base dir and other configs as needed)
- Train an LM: `python src/train_lm.py data.base_dir=XXX tokenizer.path=models/tokenizer/word-level.json`