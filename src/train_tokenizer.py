import hydra
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer


@hydra.main(config_path="../config", config_name="train-tokenizer", version_base=None)
def train_tokenizer(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = hydra.utils.instantiate(cfg.tokenizer.model)
    tokenizer = Tokenizer(model)
    # some tokenizers require a pre_tokenizer, some do not
    if "pre_tokenizer" in cfg.tokenizer:
        tokenizer.pre_tokenizer = hydra.utils.instantiate(cfg.tokenizer.pre_tokenizer)
    trainer = hydra.utils.instantiate(cfg.tokenizer.trainer, _convert_="object")
    # pass raw text files (splits) to the trainer
    tokenizer.train([cfg.data.splits[split] for split in cfg.data.splits], trainer)
    tokenizer.save(cfg.output_file)


if __name__ == "__main__":
    train_tokenizer()
