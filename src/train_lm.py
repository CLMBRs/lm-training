"""
Primary script for using HuggingFace Trainer to train a language model.
"""

import hydra
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerFast


@hydra.main(config_path="../config", config_name="train-lm", version_base=None)
def train_lm(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    ds_dict: DatasetDict = hydra.utils.instantiate(cfg.dataset, _convert_="object")
    tokenizer: PreTrainedTokenizerFast = hydra.utils.instantiate(cfg.tokenizer)
    # tokenize the datasets!
    ds_dict = ds_dict.map(lambda examples: tokenizer(examples["text"]), batched=True)

    train_ds: Dataset = ds_dict[cfg.train_split]
    eval_ds: Dataset = ds_dict[cfg.eval_split]
    trainer = hydra.utils.instantiate(
        cfg.trainer, train_dataset=train_ds, eval_dataset=eval_ds, _convert_="object"
    )
    print(cfg)
    print(trainer)
    print(trainer.model)
    trainer.train()


if __name__ == "__main__":
    train_lm()
