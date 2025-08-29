from datasets import Dataset, DatasetDict, load_dataset

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

def pack_dataset(dataset: Dataset, block_size: int, max_len: int, eos_token_id: int, log):
    if max_len % block_size != 0:
        max_len = (max_len // block_size) * block_size
        log.info(f"Recalculating dataset length to fit block size. It is now {max_len}.")
    
    new_examples = []

    example = 0
    index = 0
    force_exit = False

    while len(new_examples) < max_len // block_size and not force_exit:
        curr_example = dataset["input_ids"][example][index:index + block_size]
        if len(curr_example) == block_size:
            index += block_size
        added_loop = 1
        while len(curr_example) < block_size:
            curr_example += [eos_token_id]
            example += 1
            if example == len(dataset["input_ids"]):
                force_exit = True
                log.info(f"Dataset is only of size {len(new_examples) * block_size}.")
                break
            if len(curr_example) == block_size:
                break
            index = block_size - len(curr_example) - 1

            curr_example += dataset["input_ids"][example][:index]
            added_loop += 1

        print(curr_example)

        if not force_exit:
            new_examples.append(curr_example)
    

    return Dataset.from_dict({
        "input_ids": new_examples,
    })


def dataset_iterator(dataset, batch_size, split, text_column):
    """Yield batches of lines from the dataset."""
    for i in range(0, len(dataset), batch_size):
        yield dataset[split][i : i + batch_size][text_column]