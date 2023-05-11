"""
Primary script for using HuggingFace Trainer to train a language model.
"""

from datasets import Dataset, DatasetDict
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_lm(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    ds_dict: DatasetDict = hydra.utils.instantiate(cfg.dataset)
    train_ds: Dataset = ds_dict[cfg.train_split]
    eval_ds: Dataset = ds_dict[cfg.eval_split]
    cfg = hydra.utils.instantiate(
        cfg, trainer={"train_dataset": train_ds, "eval_dataset": eval_ds}
    )
    print(cfg)
    print(cfg.trainer.model)


if __name__ == "__main__":
    train_lm()
