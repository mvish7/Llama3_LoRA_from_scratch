import os
import time

import lightning as L
import numpy as np
import torch

from llama3_2_lora.model import Llama3Model
from llama3_2_lora.model.llama_3_2_utils import (
    load_weights_into_llama,
    LLAMA32_3B_CONFIG,
    LLAMA32_1B_CONFIG,
    load_ckpt,
    rescale_theta,
    model_memory_size,
)
from llama3_2_lora.lora.lora_utils import inject_lora_layers, lora_state_dict


instruction_tuning = True
eval_interval = 100
save_interval = 100
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 2
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 50000 * 3 // micro_batch_size
weight_decay = 0.0
max_seq_length = 128
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100


def build_model(
    ckpt_path: str,
):
    """
    instantiates the model
    """

    old_context_length = LLAMA32_3B_CONFIG["context_length"]
    LLAMA32_3B_CONFIG["context_length"] = 4096

    LLAMA32_3B_CONFIG["rope_base"] = rescale_theta(
        LLAMA32_3B_CONFIG["rope_base"],
        old_context_length,
        LLAMA32_3B_CONFIG["context_length"],
    )

    model = Llama3Model(LLAMA32_3B_CONFIG)
    ckpt_dict = load_ckpt(ckpt_path)
    model = load_weights_into_llama(model, LLAMA32_3B_CONFIG, ckpt_dict)
    return model


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [torch.tensor(data[i]["input_ids"]).type(torch.int64) for i in ix]
    labels = [torch.tensor(data[i]["labels"]).type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    if max_len > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
    )
    return loss


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    model.train()
    return out.item()


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
) -> None:
    """
    The training loop.
    """
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        fabric.backward(loss)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            torch.cuda.empty_cache()

            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                checkpoint = lora_state_dict(model)
                fabric.save(
                    os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint
                )

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms"
            )


def main(
    device: torch.device,
    ckpt_path: str,
    tokenizer_path: str,
    data_dir: str,
    out_dir: str,
):
    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir)

    model = build_model(ckpt_path)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    print(
        f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB"
    )
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

    model = inject_lora_layers(model, rank=4, alpha=1, dropout=0.4)
    # model = model.to(torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


if __name__ == "__main__":
    ckpt_path = "llama3_2_lora/checkpoint/Llama-3.2-3B-Instruct"
    tokenizer_path = "llama3_2_lora/checkpoint/Llama-3.2-3B-Instruct/original"
    data_path = "llama3_2_lora/data/"
    out_path = "llama3_2_lora/checkpoint/lora_finetuned"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    main(device, ckpt_path, tokenizer_path, data_path, out_path)
