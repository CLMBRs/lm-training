from datasets import DatasetDict, load_dataset

def load_brainscore_dataset(gen_eval, eval_split=None, seed=0, **kwargs):
    dataset_dict = load_dataset(**kwargs)
    if gen_eval:
        if eval_split is None:
            raise ValueError("Eval split must be named in gen_eval is true")
        
        split = dataset_dict["train"].train_test_split(test_size=0.2, seed=seed)

        # Create a new DatasetDict with both splits
        dataset_dict = DatasetDict({
            "train": split["train"],
            eval_split: split["test"]
        })

    return dataset_dict

def pack_dataset_gen(block_size, max_len, log):
    if max_len % block_size != 0:
        max_len = (max_len // block_size) * block_size
        log.info(f"Recalculating dataset length to fit block size. It is now {max_len}.")

    def pack(examples):
        print(examples)
        exit()
        concatenated = sum(examples["input_ids"], [])
        total_length = min((len(concatenated) // block_size) * block_size, max_len)
        return {"input_ids": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]}

    return pack

def dataset_iterator(dataset, batch_size, split, text_column):
    """Yield batches of lines from the dataset."""
    for i in range(0, len(dataset), batch_size):
        yield dataset[split][i : i + batch_size][text_column]