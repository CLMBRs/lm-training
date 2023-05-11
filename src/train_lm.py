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
    # TODO: stream-line this?
    # TODO: unify with LSTM
    model_config = hydra.utils.instantiate(cfg.model)
    model_config.vocab_size = tokenizer.get_vocab_size()
    print(model_config)
    # instantiate model
    model = hydra.utils.instantiate(cfg.model_class, model_config)
    print(model)
    print(f"Num params: {model.num_parameters()}")


if __name__ == "__main__":
    train_lm()
