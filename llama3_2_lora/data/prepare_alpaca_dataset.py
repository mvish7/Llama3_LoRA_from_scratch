"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
import torch
import json
import copy
from jsonargparse import CLI
from torch.utils.data import random_split
from tqdm import tqdm

from llama3_2_lora.model.llama_3_2_tokenizer import Tokenizer

IGNORE_INDEX = -1


def prepare(
    alpaca_filepath: str,
    test_split_size: int = 2000,
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    tokenizer_path: str = "",
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and validation dataset saved as `train.pt`
    and `val.pt`, which stores the preprocessed and tokenized
    prompts and labels.
    """

    # read the dataset
    with open(alpaca_filepath, "r") as file:
        data = json.load(file)

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data,
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("setup tokenizer")
    tokenizer = Tokenizer(model_path=tokenizer_path)

    print("Processing train split ...")
    train_set = [
        prepare_sample(sample, tokenizer, max_seq_length, mask_inputs)
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, os.path.join(os.path.dirname(alpaca_filepath), "train.pt"))

    print("Processing test split ...")
    test_set = [
        prepare_sample(sample, tokenizer, max_seq_length, mask_inputs)
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, os.path.join(os.path.dirname(alpaca_filepath), "test.pt"))


def prepare_sample(
    example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True
):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The input text is formed as a single message including all
    the instruction, the input (optional) and the response.
    The label/target is the same message but can optionally have the instruction + input text
    masked out (mask_inputs=True).

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(
        tokenizer, full_prompt, max_length=max_length, eos=False
    )
    encoded_full_prompt_and_response = tokenize(
        tokenizer, full_prompt_and_response, max_length=max_length, eos=True
    )

    # The labels are the full prompt with response,
    # but with the prompt masked out
    labels = encoded_full_prompt_and_response.copy()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = IGNORE_INDEX

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def tokenize(
    tokenizer: Tokenizer,
    string: str,
    max_length: int,
    eos=True,
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_len=max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    CLI(prepare)
