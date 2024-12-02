# Implementing LoRA from Scratch
This repo finetunes Llama3.2 models on Alpaca dataset with a custom, simplified implementation of LoRA. 

## How to use:

### Getting started:
Reading [LoRA's paper](https://arxiv.org/abs/2106.09685)\

### LoRA Toy example:
Use the `lora_toy_example.ipynb` notebook to understand need of LoRA and it's basic building blocks.

### Diving deep:
To use my customized implementation of LoRA, please follow below-mentioned steps:

* Download alpaca dataset from [here](https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json)
* Prepare the dataset using `llama3_2_lora/data/prepare_alpaca_dataset.py` script.
* Download the Llama3.2 checkpoint from [Huggingface](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
* (Optional) Understand the model architecture from `llama3_2_lora/model/llama3_2_standalone.py`
* Understand the custom LoRA implementation from `llama3_2_lora/lora/lora.py`
* Finetune the model using `llama3_2_lora/finetune_lora.py`

## ToDo:
Add finetuning with QLoRA.

