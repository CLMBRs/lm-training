"""
Primary script for using HuggingFace Trainer to train a language model.
"""

import hydra
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast


@hydra.main(config_path="../config", config_name="train-lm", version_base=None)
def train_lm(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    ds_dict: DatasetDict = hydra.utils.instantiate(cfg.dataset, _convert_="object")
    tokenizer: PreTrainedTokenizerFast = hydra.utils.instantiate(cfg.tokenizer)
    # tokenize the datasets!
    ds_dict = ds_dict.map(
        lambda examples: tokenizer(examples["text"], padding=True),
        batched=True,
    )
    # get train and val splits to feed to trainer
    train_ds: Dataset = ds_dict[cfg.train_split]
    eval_ds: Dataset = ds_dict[cfg.eval_split]
    # data collator will generate labels for language modeling
    # whicih will tell the model to return a loss, as needed for trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        _convert_="object",
    )
    trainer.train()
    trainer.save_model(output_dir=f"{trainer.args.output_dir}/best_model")


if __name__ == "__main__":
    train_lm()
