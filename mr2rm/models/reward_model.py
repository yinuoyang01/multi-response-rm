"""
Multi-response reward model: score multiple responses in one pass.

Given hidden states from the base model, take the hidden state at each response
end token, apply a scalar value head, and produce one score per response.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class MultiResponseRewardModel(nn.Module):
    """
    Args:
        base_model: A Molmo2 model loaded via AutoModelForImageTextToText (trust_remote_code).
        hidden_size: Optional override. If None, auto-detected from base_model config.
        scale_dot: If True, scale scores by sqrt(hidden_size).
    """

    # Supported activation functions for MLP value head
    SUPPORTED_ACTIVATIONS = {
        "selu": nn.SELU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        base_model: PreTrainedModel,
        hidden_size: Optional[int] = None,
        scale_dot: bool = True,
        value_head_type: str = "linear",
        value_head_hidden_dim: Optional[int] = None,
        value_head_activation: str = "selu",
        resp_repr_mode: str = "last",
    ):
        super().__init__()
        self.base_model = base_model
        self.resp_repr_mode = resp_repr_mode

        if hidden_size is None:
            if hasattr(base_model.config, "hidden_size"):
                hidden_size = base_model.config.hidden_size
            elif hasattr(base_model.config, "text_config") and hasattr(
                base_model.config.text_config, "hidden_size"
            ):
                hidden_size = base_model.config.text_config.hidden_size
            else:
                raise ValueError("Cannot determine hidden_size from base model config")
        self.hidden_size = hidden_size
        self.scale_dot = scale_dot
        self.scale = hidden_size**0.5 if scale_dot else 1.0
        self.value_head_type = value_head_type
        self.value_head_hidden_dim = value_head_hidden_dim
        self.value_head_activation = value_head_activation.lower()
        # concat doubles the input dimensionality; others stay at hidden_size
        self.value_head_input_dim = hidden_size * 2 if resp_repr_mode == "first_last_concat" else hidden_size

        if value_head_type == "linear":
            self.value_head = nn.Linear(self.value_head_input_dim, 1)
        elif value_head_type == "mlp":
            hidden_dim = value_head_hidden_dim or self.value_head_input_dim
            # Get activation function
            if self.value_head_activation not in self.SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"Unsupported activation: {self.value_head_activation}. "
                    f"Supported: {list(self.SUPPORTED_ACTIVATIONS.keys())}"
                )
            activation_cls = self.SUPPORTED_ACTIVATIONS[self.value_head_activation]
            self.value_head = nn.Sequential(
                nn.Linear(self.value_head_input_dim, hidden_dim),
                activation_cls(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported value_head_type: {value_head_type}")

        # Initialize value head weights
        self._init_value_head()

        # Match value head dtype to base model dtype
        base_dtype = next(base_model.parameters()).dtype
        self.value_head = self.value_head.to(base_dtype)
    
    def _init_value_head(self):
        """Initialize value head weights with small random values."""
        if self.value_head_type == "linear":
            # Initialize with small random values for stable training
            nn.init.normal_(self.value_head.weight, std=0.01)
            if self.value_head.bias is not None:
                nn.init.zeros_(self.value_head.bias)
        elif self.value_head_type == "mlp":
            # Initialize both linear layers in the MLP
            # First layer: hidden_size -> hidden_dim
            nn.init.normal_(self.value_head[0].weight, std=0.01)
            if self.value_head[0].bias is not None:
                nn.init.zeros_(self.value_head[0].bias)
            # Second layer: hidden_dim -> 1
            nn.init.normal_(self.value_head[2].weight, std=0.01)
            if self.value_head[2].bias is not None:
                nn.init.zeros_(self.value_head[2].bias)


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        resp_indices: Optional[torch.LongTensor] = None,
        resp_start_indices: Optional[torch.LongTensor] = None,
        rankings: Optional[torch.LongTensor] = None,  # [B, R] ground truth rankings (1=best, higher=worse, -1=invalid)
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (optional)
            resp_indices: [batch, num_responses] indices for each response end token
            **kwargs: multimodal inputs forwarded to base_model (pixel_values, token_type_ids, etc.)

        Returns:
            (scores,) where scores shape is [batch, num_responses]
        """
        if resp_indices is None:
            raise ValueError("resp_indices is required to locate response tokens")
        if self.resp_repr_mode != "last" and self.resp_repr_mode != "first":
            # modes that need both start and end
            if resp_start_indices is None:
                raise ValueError(f"resp_start_indices is required for resp_repr_mode={self.resp_repr_mode}")

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )

        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]  # [B, L, H]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state  # [B, L, H]
        else:
            raise ValueError("Model outputs must include hidden states")

        batch_size, seq_len, hidden = hidden_states.shape
        if hidden != self.hidden_size:
            self.hidden_size = hidden
            if self.scale_dot:
                self.scale = hidden**0.5

        if resp_indices.dim() != 2:
            raise ValueError("resp_indices must have shape [batch, num_responses]")
        
        # Warn if indices are out of range, but clamp them instead of raising error
        if resp_indices.min() < -1 or resp_indices.max() >= seq_len:
            import warnings
            warnings.warn(
                f"resp_indices out of range [-1, {seq_len-1}]: "
                f"min={resp_indices.min().item()}, max={resp_indices.max().item()}, "
                f"seq_len={seq_len}. Clamping to valid range."
            )

        resp_mask = resp_indices >= 0  # [B, R]
        # IMPORTANT: torch.take_along_dim does NOT support negative indices!
        # Replace -1 (invalid/padding) with 0 for gather operation.
        # The result will be masked out later using resp_mask.
        # For valid indices (>= 0), clamp to [0, seq_len-1]
        # For invalid indices (-1), use 0 as placeholder (will be masked out)
        safe_resp_indices = torch.where(
            resp_indices >= 0,
            resp_indices.clamp(min=0, max=seq_len - 1),
            torch.zeros_like(resp_indices)  # Use 0 instead of -1 for gather
        )
        resp_vecs = torch.take_along_dim(
            hidden_states,
            safe_resp_indices[..., None].expand(-1, -1, hidden_states.size(-1)),
            dim=1,
        )  # [B, R, H]

        start_vecs = None
        start_mask = resp_mask
        if resp_start_indices is not None:
            if resp_start_indices.dim() != 2:
                raise ValueError("resp_start_indices must have shape [batch, num_responses]")
            if resp_start_indices.min() < -1 or resp_start_indices.max() >= seq_len:
                import warnings
                warnings.warn(
                    f"resp_start_indices out of range [-1, {seq_len-1}]: "
                    f"min={resp_start_indices.min().item()}, max={resp_start_indices.max().item()}, "
                    f"seq_len={seq_len}. Clamping to valid range."
                )
            start_mask = resp_start_indices >= 0
            safe_start_indices = torch.where(
                resp_start_indices >= 0,
                resp_start_indices.clamp(min=0, max=seq_len - 1),
                torch.zeros_like(resp_start_indices),
            )
            start_vecs = torch.take_along_dim(
                hidden_states,
                safe_start_indices[..., None].expand(-1, -1, hidden_states.size(-1)),
                dim=1,
            )

        # Build response representation according to the selected mode
        if self.resp_repr_mode == "last":
            # Use only the last token's hidden state (default, original behavior)
            # Output shape: [B, R, H]
            resp_repr = resp_vecs
            valid_mask = resp_mask
        elif self.resp_repr_mode == "first":
            # Use only the first token's hidden state
            # Output shape: [B, R, H]
            if start_vecs is None:
                raise ValueError("resp_start_indices required when resp_repr_mode='first'")
            resp_repr = start_vecs
            valid_mask = start_mask
        elif self.resp_repr_mode == "first_last_concat":
            # Concatenate first and last token hidden states along feature dimension
            # Preserves all information from both tokens, doubles input dimension to value head
            # Output shape: [B, R, 2*H] (value head input_dim = 2*H)
            if start_vecs is None:
                raise ValueError("resp_start_indices required when resp_repr_mode='first_last_concat'")
            resp_repr = torch.cat([start_vecs, resp_vecs], dim=-1)
            valid_mask = resp_mask & start_mask
        elif self.resp_repr_mode == "first_last_add":
            # Element-wise addition of first and last token hidden states
            # Fuses information linearly, keeps same dimension as single token
            # Output shape: [B, R, H] (value head input_dim = H)
            if start_vecs is None:
                raise ValueError("resp_start_indices required when resp_repr_mode='first_last_add'")
            resp_repr = start_vecs + resp_vecs
            valid_mask = resp_mask & start_mask
        elif self.resp_repr_mode == "first_last_sub":
            # Element-wise subtraction: last - first token hidden states
            # Captures the "change" or "delta" from start to end of response
            # Output shape: [B, R, H] (value head input_dim = H)
            if start_vecs is None:
                raise ValueError("resp_start_indices required when resp_repr_mode='first_last_sub'")
            resp_repr = resp_vecs - start_vecs
            valid_mask = resp_mask & start_mask
        elif self.resp_repr_mode == "response_mean":
            # Average all tokens in the response (from start to end inclusive)
            # This follows the approach in arXiv:2501.12368 for reward modeling
            # Output shape: [B, R, H] (value head input_dim = H)
            if resp_start_indices is None:
                raise ValueError("resp_start_indices required when resp_repr_mode='response_mean'")

            # Compute mean over all tokens from start to end for each response
            # hidden_states: [B, L, H], resp_start_indices: [B, R], resp_indices (end): [B, R]
            batch_size_r, num_responses = resp_indices.shape
            device = hidden_states.device
            dtype = hidden_states.dtype

            resp_repr_list = []
            for b in range(batch_size_r):
                response_means = []
                for r in range(num_responses):
                    start_idx_val = resp_start_indices[b, r].item()
                    end_idx_val = resp_indices[b, r].item()

                    # Check if this response is valid (not masked with -1)
                    if start_idx_val < 0 or end_idx_val < 0 or start_idx_val > end_idx_val:
                        # Invalid response: use zeros (will be masked out later)
                        response_means.append(torch.zeros(hidden_states.size(-1), device=device, dtype=dtype))
                    else:
                        # Extract all tokens from start to end (inclusive) and compute mean
                        # Clamp indices to valid range
                        start_idx_val = max(0, min(start_idx_val, seq_len - 1))
                        end_idx_val = max(0, min(end_idx_val, seq_len - 1))
                        response_tokens = hidden_states[b, start_idx_val:end_idx_val + 1, :]  # [T, H]
                        response_mean = response_tokens.mean(dim=0)  # [H]
                        response_means.append(response_mean)

                # Stack responses for this batch: [R, H]
                resp_repr_list.append(torch.stack(response_means, dim=0))

            # Stack all batches: [B, R, H]
            resp_repr = torch.stack(resp_repr_list, dim=0)
            valid_mask = resp_mask & start_mask
        else:
            raise ValueError(f"Unsupported resp_repr_mode: {self.resp_repr_mode}")

        scores = self.value_head(resp_repr).squeeze(-1)  # [B, R]
        if self.scale_dot and self.scale != 0:
            scores = scores / self.scale

        # Mask out invalid responses
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        return (scores,)

