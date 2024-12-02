import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from safetensors.torch import load_file

LLAMA32_3B_CONFIG = {
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 131_072,  # Context length
    "emb_dim": 3072,  # Embedding dimension
    "n_heads": 24,  # Number of attention heads
    "n_layers": 28,  # Number of layers
    "hidden_dim": 8192,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to save memory
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}

LLAMA32_1B_CONFIG = {
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 131_072,  # Context length
    "emb_dim": 2048,  # Embedding dimension
    "n_heads": 32,  # Number of attention heads
    "n_layers": 16,  # Number of layers
    "hidden_dim": 8192,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to save memory
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}


def download_ckpt():
    """
    download the checkpoint and tokenizer from huggingface
    """
    login()
    _ = hf_hub_download(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        filename="original/tokenizer.model",
        local_dir="Llama-3.2-3B-Instruct",
    )

    for i in range(1, 3):
        _ = hf_hub_download(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            filename="model-0000{i}-of-00002.safetensors",
            local_dir="Llama-3.2-3B-Instruct",
        )


def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
        )

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params) -> nn.Module:
    """
    Loads pre-trained weights into the LLaMA model.

    Args:
    - model (nn.Module): The LLaMA model instance to load weights into.
    - param_config (dict): A dictionary containing configuration parameters
        for the model. Specifically, it should contain the value of 'n_layers'
        which represents the number of layers in the model.
    - params (dict): A dictionary containing the pre-trained weights
    and their corresponding layer names.

    Returns:
    None
    """
    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        params["model.embed_tokens.weight"],
        "model.embed_tokens.weight",
    )

    for l in range(param_config["n_layers"]):

        # Load attention weights
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight",
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight",
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight",
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight",
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight",
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight",
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight",
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight",
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight",
        )

    # Load output layer weights
    model.final_norm.weight = assign(
        model.final_norm.weight, params["model.norm.weight"], "model.norm.weight"
    )

    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(
            model.out_head.weight, params["lm_head.weight"], "lm_head.weight"
        )
    else:
        model.out_head.weight = assign(
            model.out_head.weight,
            params["model.embed_tokens.weight"],
            "model.embed_tokens.weight",
        )
        print("Model uses weight tying.")

    return model


def load_ckpt(ckpt_path):
    """
    Loads weights from multiple checkpoint files in a specified directory.

    This function iterates over all files in the provided
    checkpoint path that end with "safetensors",
    loads each file, and updates a combined dictionary of
    weights. The loaded weights are stored in the
    `combined_weights` variable.

    Args:
        ckpt_path (str): The path to the directory
        containing the checkpoint files.

    Returns:
        dict: A dictionary containing all the
        combined weights from the individual checkpoint files.
    """
    combined_weights = {}
    ckf_files = [f for f in os.listdir(ckpt_path) if f.endswith("safetensors")]
    for ckf in ckf_files:
        curr_weights = load_file(os.path.join(ckpt_path, ckf))
        combined_weights.update(curr_weights)
    return combined_weights


def rescale_theta(theta_old, context_length_old, context_length_new):
    """
    rescale RoPE theta to make llama3.2 model managable on single desktop
    """
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new


def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb
