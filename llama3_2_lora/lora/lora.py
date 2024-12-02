import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    A LoRA module that augments an existing linear layer with low-rank updates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        # Handle transposed weight formats
        fan_in_fan_out: bool = False,
        # Merge LoRA weights into base weights during inference
        merge_weights: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False  # Track whether weights are merged

        # Base (frozen) weights
        # Placeholder for pre-trained weights -- will be modified when
        # injecting lora into network
        self.weight = None

        # Trainable LoRA parameters
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def set_base_weights(self, weight):
        """
        Sets the base (pre-trained) weights for the layer and freezes them.
        """
        self.weight = nn.Parameter(weight)
        self.weight.requires_grad = False  # Freeze base weights
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)  # Transpose if needed

    def merge(self):
        """
        Merge LoRA weights into the base weights for inference efficiency.
        """
        if not self.merged and self.rank > 0:
            T = lambda w: w.transpose(0, 1) if self.fan_in_fan_out else w
            self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        """
        Unmerge LoRA weights from the base weights (useful for training).
        """
        if self.merged and self.rank > 0:
            T = lambda w: w.transpose(0, 1) if self.fan_in_fan_out else w
            self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def forward(self, x):
        """
        Forward pass with optional merging of weights.
        """
        assert (
            self.weight is not None
        ), "Base weights must be set using `set_base_weights`."

        T = lambda w: w.transpose(0, 1) if self.fan_in_fan_out else w

        # If LoRA is merged, use the base weight directly
        if self.merged or self.rank == 0:
            return F.linear(x, T(self.weight))

        # Base projection + LoRA update
        result = F.linear(x, T(self.weight))
        lora_update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result += self.scaling * lora_update
        return result
