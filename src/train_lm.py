"""
Primary script for using HuggingFace Trainer to train a language model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_lm(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    train_lm()
