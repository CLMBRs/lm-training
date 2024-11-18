"""Primary script for using HuggingFace Trainer to train a language model.

For examples of how to run, see README.md.
"""

from contextlib import ExitStack
import logging
import os
import sys

from datasets import Dataset, DatasetDict
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
from transformers.integrations import is_wandb_available
from transformers.trainer_utils import get_last_checkpoint
import numpy as np

log = logging.getLogger(__name__)

# resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])


@hydra.main(config_path="../config", config_name="train-lm", version_base=None)
def train_lm(cfg: DictConfig) -> None:
    """
    Config fields:

    trainer:
        args (*required*):
            output_dir (*required*):
                .

            If not provided, defaults to a new `transformers.TrainingArguments` instance
            with `output_dir` set to "./" (the working directory).
    training_args:
        This should instantiate into a `transformers.TrainingArguments` object. It will
        be passed to the `Trainer` constructor as its `args` parameter.

    dynamic_resume (bool, *optional*):
        If true:
            * Will resume a previous job or start a new one in the absence of a
            previous job.
            * trainer.args.output_dir must *uniquely* identify each job and *uniquely*
            store the checkpoints associated with the job across runs. If wandb is
            enabled, this will also be where a file is persisted containing the unique
            wandb run associated with the run, allowing preempted or otherwise
            prematurely terminated jobs to continue logging where they left off upon
            resuming.
            * A best practice is to place trainer.args.output_dir within
            hydra.runtime.output_dir, though this is not strictly necessary, and to
            construct hydra.runtime.output_dir out of core  parameters of the given
            training job via config interpolation

            * trainer.args.report_to is set to all or contains
            wandb, a unique wandb run ID will be persisted to
            (hydra.runtime.output_dir)/.wandb_run_id hydra.job.chdir must be True (the default)This directory will store the
            corresponding wandb run ID,  which will then be used in future runs if the
            job is preempted or otherwise exits prematurely.
    dynamic_resume (bool, *optional*):
        If set to true, then hydra.runtime.output_dir must *uniquely* identify each job
        across runs. Further, trainer.args.output_dir must This directory will store the corresponding wandb run ID,  which
        will then be used in future runs if the job is preempted or otherwise exits
        prematurely.
    """
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    ds_dict: DatasetDict = hydra.utils.instantiate(cfg.dataset, _convert_="object")
    tokenizer: PreTrainedTokenizerFast = hydra.utils.instantiate(cfg.tokenizer)
    if not cfg.data.get("is_tokenized", False):
        log.info("Tokenizing dataset.")
        ds_dict = ds_dict.map(
            lambda examples: tokenizer(
                examples[cfg.text_field], padding=True, truncation=True
            ),
            batched=True,
        )
    else:
        log.info("Dataset is already tokenized; proceeding.")
    # get train and val splits to feed to trainer
    train_ds: Dataset = ds_dict[cfg.train_split]
    eval_ds: Dataset = ds_dict[cfg.eval_split]
    if "args" not in cfg.trainer:
        log.error("Must provide an args field in the trainer config.")
        sys.exit(1)
    # if "output_dir" not in cfg.trainer.args:
    #     log.error("Must provide an output_dir field in the trainer.args config.")
    #     sys.exit(1)

    # We create a dummy TrainingArguments purely to make the code simpler, as an actual
    # TrainingArguments instance will have default values filled in for missing
    # fields, whereas an OmegaConfig will not. Thus we instantiate a temporary
    # TrainingArguments object and fill specific values we need from it back into the
    # original config, we that we do not have to check if the value exists in the config
    # every time we reference it. We cannot instead directly pass this object to the
    # Trainer instantiate call near the end of this function because Hydra interprets
    # dataclasses as (structured) config, so they can't be used as passthrough arguments
    # to instantiate (TrainingArguments is a dataclass)
    with open_dict(cfg):
        training_args_tmp: TrainingArguments = (
            hydra.utils.instantiate(cfg.trainer.args, _convert_="object")
            if "args" in cfg.trainer
            else TrainingArguments(output_dir="./")
        )

        cfg.trainer.args.update(
            {
                "dataloader_num_workers": training_args_tmp.dataloader_num_workers,
                "max_steps": training_args_tmp.max_steps,
                "num_train_epochs": training_args_tmp.num_train_epochs,
                "output_dir": training_args_tmp.output_dir,
                "per_device_train_batch_size": training_args_tmp.per_device_train_batch_size,
                "report_to": training_args_tmp.report_to,
            }
        )

    if cfg.get("use_iterable_dataset", False):
        log.info("Converting datasets to iterable.")
        # LR scheduler requires max_steps with iterable datasets because they lack
        # __len__ function, so we dynamically calculate it here if not provided
        if cfg.trainer.args.max_steps < 0:  # HF default: -1
            cfg.trainer.args.max_steps = int(
                cfg.trainer.args.num_train_epochs
                * (len(train_ds) / cfg.trainer.args.per_device_train_batch_size)
            )
        train_shards = (
            cfg.get("train_shards_per_worker", 8)
            * cfg.trainer.args.dataloader_num_workers
        )
        eval_shards = (
            cfg.get("eval_shards_per_worker", 8)
            * cfg.trainer.args.dataloader_num_workers
        )
        train_ds = train_ds.to_iterable_dataset(num_shards=train_shards)
        eval_ds = eval_ds.to_iterable_dataset(num_shards=eval_shards)

    # data collator will generate labels for language modeling
    # which will tell the model to return a loss, as needed for trainer
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collator_config = cfg.collator
    collator_config_with_tokenizer = {**collator_config, 'tokenizer': tokenizer}
    print(f"Data Collator settings: {collator_config_with_tokenizer}")
    
    with ExitStack() as stack:
        if "dynamic_resume" in cfg and cfg.dynamic_resume:
            log.info("Dynamic resume enabled.")
            # checkpoints directory won't exist on first train run, so create it
            os.makedirs(cfg.trainer.args.output_dir, exist_ok=True)
            if get_last_checkpoint(cfg.trainer.args.output_dir):
                resume_from_checkpoint = True
            else:  # no checkpoints exist
                resume_from_checkpoint = False
            # transformers checks to see if wandb is enabled and installed; if not,
            # report_to won't contain wandb even if it was provided to TrainingArguments
            if (
                (
                    hasattr(cfg.trainer.args.report_to, "__contains__")
                    and type(cfg.trainer.args.report_to is not str)
                )
                and "wandb" in cfg.trainer.args.report_to
            ) or cfg.trainer.args.report_to == "wandb":
                import wandb

                wandb_run_id_file = os.path.join(
                    cfg.trainer.args.output_dir, ".wandb_run_id"
                )

                if os.path.isfile(wandb_run_id_file):
                    with open(wandb_run_id_file, "r") as f:
                        wandb_run_id = f.read()
                    log.info(f"Resuming wandb run with ID: {wandb_run_id}")
                else:
                    with open(wandb_run_id_file, "w") as f:
                        wandb_run_id = wandb.util.generate_id()
                        f.write(wandb_run_id)
                    log.info(f"Starting new wandb run with ID: {wandb_run_id}")
                # if using dynamic_resume, must initialize wandb manually; otherwise, we
                # can let Trainer do it when its constructor is called
                stack.enter_context(
                    wandb.init(
                        id=wandb_run_id,
                        resume="allow",
                        # other arguments to init should be set via environment
                        # variables, e.g. via hydra.job.env_set
                    )
                )
        else:
            resume_from_checkpoint = False

        # Now, the core training setup

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            # args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator_config_with_tokenizer,
            _convert_="object",
        )

        log.info(f"Final TrainingArguments:\n{trainer.args}")

        model_parameters = filter(lambda p: p.requires_grad, trainer.model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])

        log.info(f"Number of model params: {num_params}")

        log.info(f"resume_from_checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(
            output_dir=f"{trainer.args.output_dir}/"
            f"{'best_model' if trainer.args.load_best_model_at_end else 'last_model'}"
        )


if __name__ == "__main__":
    train_lm()
