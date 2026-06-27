import torch
import torch.nn as nn
from transformers.integrations import use_kernel_forward_from_hub


@use_kernel_forward_from_hub("RMSNorm")
class YaeRMSNorm(nn.Module):
    """
    Ref: https://docs.pytorch.org/docs/2.12/generated/torch.nn.RMSNorm.html
    RMSNorm (Root Mean Square Normalization)
    A normalization architecture for the Qwen3 model family (equivalent to T5LayerNorm).

    ---
    🧐 What is RMSNorm? (And how does it differ from LayerNorm?)
    - An evolution of the traditional LayerNorm used in earlier model architectures.
    - Standard LayerNorm: Computes both Mean (centering) and Variance (scaling).
    - RMSNorm: Skips the Mean calculation! It only calculates the Root Mean Square
      of the features.
    - Advantages: Improves computational efficiency by 7% - 64% per layer. This is why
      modern LLMs (like LLaMA and Qwen) prefer RMSNorm—it saves time and boosts
      training speed significantly.

    ---
    🚀 Decorator: @use_kernel_forward_from_hub("RMSNorm")
    - Purpose: Pulls custom CUDA/Triton kernels for RMSNorm from the Hub to replace
      the default forward pass.
    - Why?: Pure PyTorch code executes in Eager mode (step-by-step). Utilizing
      specialized kernels fuses the operations directly on the GPU, achieving
      maximum performance and extreme speed.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        Initializes the structure and registers learnable parameters.

        Args:
            hidden_size (int): The feature dimension size.
            eps (float): Small epsilon value to prevent division-by-zero errors.

        Attributes:
            self.weight (nn.Parameter): Learnable gain parameter (γ), initialized to ones.
            self.variance_epsilon (float): Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Core math logic for RMSNorm.

        Step-by-step processing:
        1. Cast input to float32 to ensure numerical precision during calculation.
        2. pow(2): Square all elements in the tensor.
        3. mean(-1, keepdim=True): Compute the mean across the feature dimension.
        4. torch.rsqrt(): Efficiently computes 1 / sqrt(x + eps).
        5. Scale and restore the original data type before applying the learnable weight.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        """
        Returns a clean string representation for model structure printing.
        Example: YaeRMSNorm((4096,), eps=1e-06)
        """
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


if __name__ == "__main__":
    # 1. Simulate input data
    torch.manual_seed(42)
    sample_input = torch.randn(1, 2, 4) * 10 + 5

    print("=== 📥 1. Original Input Tensor ===")
    print(sample_input)
    print("-" * 50)

    # 2. Standard PyTorch LayerNorm
    pytorch_layernorm = nn.LayerNorm(4)
    output_ln = pytorch_layernorm(sample_input)

    print("=== ⚖️ 2. Output from LayerNorm (PyTorch Standard) ===")
    print(output_ln)
    # [Yae Math Summary]: LayerNorm centers the mean to 0.
    # Values near 0.0000004 verify that the mean has been effectively removed.
    print(f"-> Mean check after LayerNorm: {output_ln.mean(-1).tolist()}")
    print("-" * 50)

    # 3. YaeRMSNorm
    yae_rmsnorm = YaeRMSNorm(hidden_size=4)
    output_rms = yae_rmsnorm(sample_input)

    print("=== 🚀 3. Output from YaeRMSNorm ===")
    print(output_rms)
    # [Yae Math Summary]: RMSNorm does not center the mean.
    # It focuses on controlling the scale (magnitude) of the values to prevent
    # exploding gradients while maximizing GPU throughput.
    print(f"-> Mean check after RMSNorm: {output_rms.mean(-1).tolist()}")
    print("-" * 50)
