"""
Primary script for using HuggingFace Trainer to train a language model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="train-lm", version_base=None)
def train_lm(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # build tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    # get config first, then instantiate model
    # this way, can set vocab size via the tokenizer
    config = hydra.utils.instantiate(cfg.model.config)
    config.vocab_size = tokenizer.get_vocab_size()
    print(config)


if __name__ == "__main__":
    train_lm()
