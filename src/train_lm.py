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
        # tokenize the text
        # for language modeling, we don't need token_type_ids, so they are not returned
        # this is hard-coded for this use case, but could be parameterized in the future
        # if there's ever a need for that
        lambda examples: tokenizer(
            examples["text"], padding=True, return_token_type_ids=False
        ),
        batched=True,
    )
    # remove columns if specified to do so
    if "remove_columns_from_tokenized_data" in cfg:
        ds_dict = ds_dict.remove_columns(cfg.remove_columns_from_tokenized_data)
    # get train and val splits to feed to trainer
    train_ds: Dataset = ds_dict[cfg.train_split]
    eval_ds: Dataset = ds_dict[cfg.eval_split]
    print(train_ds)
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


if __name__ == "__main__":
    train_lm()
