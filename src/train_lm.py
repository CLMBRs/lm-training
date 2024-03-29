"""Primary script for using HuggingFace Trainer to train a language model.

For examples of how to run, see README.md.
"""

import logging

from datasets import Dataset, DatasetDict
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
import numpy as np

log = logging.getLogger(__name__)

# resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])


@hydra.main(config_path="../config", config_name="train-lm", version_base=None)
def train_lm(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    ds_dict: DatasetDict = hydra.utils.instantiate(cfg.dataset, _convert_="object")
    tokenizer: PreTrainedTokenizerFast = hydra.utils.instantiate(cfg.tokenizer)
    # tokenize the datasets!
    ds_dict = ds_dict.map(
        lambda examples: tokenizer(examples[cfg.text_field], padding=True),
        batched=True,
    )
    # get train and val splits to feed to trainer
    train_ds: Dataset = ds_dict[cfg.train_split]
    eval_ds: Dataset = ds_dict[cfg.eval_split]
    # data collator will generate labels for language modeling
    # which will tell the model to return a loss, as needed for trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        _convert_="object",
    )

    model_parameters = filter(lambda p: p.requires_grad, trainer.model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])

    log.info(f"Number of model params: {num_params}")

    trainer.train()
    trainer.save_model(
        output_dir=f"{trainer.args.output_dir}/"
        f"{'best_model' if trainer.args.load_best_model_at_end else 'last_model'}"
    )


if __name__ == "__main__":
    train_lm()
