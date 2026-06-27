import torch
import torch.nn as nn

from typing import Optional
from collections.abc import Callable

from transformers import dynamic_rope_update
from transformers.utils.generic import maybe_autocast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack

from YaeConfig import YaeConfig


class YaeRotaryEmbedding(nn.Module):
    """
    YaeRotaryEmbedding (RoPE - Rotary Position Embedding)
    A production-ready vector rotation-based positional encoding system.

    ---
    Core Functionality:
    Creates a 'degree ruler' that informs the Transformer of each token's position
    in a sequence. Instead of simple additive positional encoding, it rotates
    vectors within a complex space, allowing the model to excel at understanding
    relative positional relationships between tokens.
    """

    inv_freq: torch.Tensor  # Linting fix for register_buffer

    def __init__(self, config: "YaeConfig", device=None):
        """
        Initialization:
        Prepares the inverse frequencies for vector rotation. Designed to be
        flexible and compatible with various RoPE variants (e.g., Default, YaRN, Llama3).
        """
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # [Fix 1] Safe retrieval using .get() to prevent errors if keys are missing in config
        rope_params = getattr(self.config, "rope_parameters", {}) or {}
        self.rope_type = rope_params.get("rope_type", "default")

        rope_init_fn: Callable = self.compute_default_rope_parameters

        # [Fix 2] Validate existence before deployment
        if self.rope_type != "default" and self.rope_type in ROPE_INIT_FUNCTIONS:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # Initialize the frequency array (inv_freq)
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        # 🔒 Registering as a buffer: Tells PyTorch this is a non-trainable constant
        # to be moved to the appropriate device (e.g., GPU).
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: "YaeConfig",
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Mathematical engine for generating base frequencies (inv_freq).

        Equation:
        $inv\_freq_i = 1.0 / (base^{(2i / dim)})$
        - base: Frequency base (typically 10,000 for standard models).
        - dim: Head dimension.
        - i: Dimension index.
        """

        # [Fix 4] Safe retrieval of base frequency
        rope_params = getattr(config, "rope_parameters", {}) or {}
        base = rope_params.get("rope_parameters", 10000.0)

        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0

        # 🧮 Compute inverse frequencies (the heart of RoPE)
        # Higher frequencies (early dims) rotate faster for local detail,
        # while lower frequencies (later dims) rotate slowly for long-range relations.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()  # 🛑 Gradients are not required for positional calculation
    @dynamic_rope_update  # 🌀 Hook for advanced RoPE types (e.g., dynamic scaling)
    def forward(self, x, position_ids):
        """
        Translates 'position' into 'rotation angle'.
        x: Input tensor (Query or Key)
        position_ids: Token position identifiers (0, 1, 2, ...)
        """

        # 1. Expand inv_freq to match position_ids for batch operations
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )

        # 2. Prepare position_ids for matrix multiplication
        position_ids_expanded = position_ids[:, None, :].float()

        # 3. Precision handling: Force float32 for angular calculation stability
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )

        with maybe_autocast(device_type=device_type, enabled=False):
            # 4. 🧮 Angular frequency: (Frequency * Position) = Theta (θ)
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)

            # 5. Create (cos, sin) pairs for 2D rotation
            # Each dimension pair requires a matching (cos, sin) vector
            emb = torch.cat((freqs, freqs), dim=-1)

            # 6. Compute Trigonometry with scaling
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # Return in the original input dtype
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input to perform 2D rotation.

    Logic:
    - RoPE transformation: x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
    - rotate_half(x) performs a 90-degree rotation (x, y) -> (-y, x)
    - Returning (-x2, x1) changes the orientation correctly for the sine component.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands the Key/Value heads to match the Query head count.

    This function is a core component of GQA (Grouped Query Attention), a standard
    technique in modern LLMs (e.g., Qwen3) to optimize memory usage and processing speed.

    Background:
    - Standard Attention: Query (Q) heads often outnumber Key/Value (K/V) heads.
    - Limitation: Q and K/V head counts must be equal for dot-product attention calculation.
    - Mechanism: This function 'broadcasts' the K/V heads to match the Q head count,
      enabling efficient multi-head attention without redundant memory allocation.

    Args:
        hidden_states: Input tensor of shape (batch, num_kv_heads, seqlen, head_dim).
        n_rep: The number of repetitions required to match the Q head count.

    Returns:
        Tensor of shape (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    # Expand dimensions to replicate K/V heads
    # Using expand() creates a virtual view, saving significant memory vs physical copy
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )

    # Flatten back to the required (batch, num_attention_heads, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    Computes Scaled Dot-Product Attention using standard PyTorch (Eager) execution.

    This function implements the standard Transformer attention mechanism:
    1. Expands K/V heads to match Query head count (GQA support).
    2. Calculates raw attention scores via matrix multiplication (Q @ K^T).
    3. Applies scaling, masking, softmax, and dropout to normalize weights.
    4. Computes the final output by aggregating values (Attention Weights @ V).

    Args:
        module: The parent attention module (containing head group configurations).
        query, key, value: Input tensors representing Q, K, and V projections.
        attention_mask: Mask tensor to ignore padding or future tokens (causal mask).
        scaling: The scaling factor (usually 1/sqrt(head_dim)) to stabilize gradients.
        dropout: Probability for dropout application during training.

    Returns:
        attn_output: The weighted sum of value states.
        attn_weights: The normalized attention scores (Softmax output).
    """

   # 1. Align K/V heads with Query heads (Grouped Query Attention)
    n_kv_groups = int(getattr(module, "num_key_value_groups", 1))
    key_states = repeat_kv(key, n_kv_groups)
    value_states = repeat_kv(value, n_kv_groups)

    # 2. Compute raw attention scores (Q * K^T) and apply scaling
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # 3. Apply mask (e.g., Causal/Padding mask) to forbid attention to specific tokens
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # 4. Normalize weights using Softmax (float32 for precision) and apply dropout
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    # 5. Compute final weighted output (Weights * V)
    attn_output = torch.matmul(attn_weights, value_states)

    # 6. Reshape output: transpose back to (batch, seqlen, num_heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
