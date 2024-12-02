import torch.nn as nn
from llama3_2_lora.lora.lora import LoRALayer


def inject_lora_layers(
    model: nn.Module,
    rank: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_layers: list = None,
    fan_in_fan_out: bool = False,
    merge_weights: bool = False,
):
    """
    Dynamically inject LoRA layers into a model by replacing `nn.Linear` layers.

    Args:
        model: Pre-trained model to modify.
        rank: Low-rank approximation rank.
        alpha: Scaling factor for LoRA.
        dropout: Dropout probability for LoRA layers.
        target_layers: List of layer names (W_key, W_query, W_value) to replace (inject LoRA).
        fan_in_fan_out: Whether to handle transposed weight formats.
        merge_weights: Merge LoRA weights into base weights for inference.
    Returns:
        The modified model with LoRA layers injected.
    """
    for name, module in model.named_children():
        # Recursively inject LoRA into child modules
        inject_lora_layers(
            module, rank, alpha, dropout, target_layers, fan_in_fan_out, merge_weights
        )

        # Replace targeted `nn.Linear` layers with LoRA-enhanced versions
        if isinstance(module, nn.Linear):
            if target_layers is None or name in target_layers:
                lora_layer = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    fan_in_fan_out=fan_in_fan_out,
                    merge_weights=merge_weights,
                )
                # Retain pre-trained weights and freeze them
                lora_layer.set_base_weights(module.weight)
                if module.bias is not None:
                    lora_layer.bias = nn.Parameter(module.bias.clone())
                setattr(model, name, lora_layer)  # Replace the layer

    return model


def lora_state_dict(model: nn.Module):
    """
    fetches weights of lora layers
    Args:
        model: lora injected model
    """

    lora_state_dict = model.state_dict()

    to_return = {}
    for k in lora_state_dict:
        if "lora_" in k:
            to_return[k] = lora_state_dict[k]
            bias_name = k.split("lora_")[0] + "bias"
            if bias_name in lora_state_dict:
                to_return[bias_name] = lora_state_dict[bias_name]
    return to_return
