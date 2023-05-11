"""
Primary script for using HuggingFace Trainer to train a language model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer
from datasets import load_dataset


@hydra.main(config_path="../config", config_name="train-lm", version_base=None)
def train_lm(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # build tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    # enable padding for training; is this the right way?!
    tokenizer.enable_padding()
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

    # build train and val datasets
    # TODO: generalize from "raw" version?
    dataset = load_dataset("text", data_files=OmegaConf.to_object(cfg.data.splits))
    print(dataset)
    print(dataset['train'])

    training_args = hydra.utils.instantiate(cfg.trainer)
    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
    )
    trainer.train()


if __name__ == "__main__":
    train_lm()
