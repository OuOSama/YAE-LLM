import torch.nn as nn
from transformers.activations import ACT2FN


class YaeMLP(nn.Module):
    """
    Ref: https://docs.pytorch.org/docs/2.12/generated/torch.nn.SiLU.html
    YaeMLP (Multi-Layer Perceptron / Feed-Forward Network)
    Processing block based on SwiGLU, inspired by the Qwen 3 architecture.

    ---
    🧐 What is SiLU?
    SiLU (Sigmoid Linear Unit), also known as Swish, is an activation function.
    The underlying mathematical formula is: f(x) = x * sigmoid(x).

    Special properties that make it superior to the legacy ReLU:
    1. Smooth & Continuous: The curve is smooth, ensuring that gradients remain
       well-behaved during backpropagation.
    2. Non-monotonicity: Features a slight dip in the negative domain, allowing some
       negative values to pass through. This enhances the model's ability to learn
       complex, deep structural features, leading to significantly higher stability
       and performance.
    ---

    🧠 SwiGLU Mathematical Mechanism:
    Instead of a single linear layer, modern models employ three parallel pipelines:
    1. gate_proj + act_fn: Acts as an "information filter" (gate) to prioritize
       essential features.
    2. up_proj: Expands the data dimensions, providing the model with a broader
       "workspace" for complex processing.
    3. down_proj: Compresses the filtered results back to the original dimension
       before output.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # Input dimension
        self.intermediate_size = config.intermediate_size

        # Parallel expansion pathways (hidden_size -> intermediate_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        # Compression pathway (intermediate_size -> hidden_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Activation function (usually "silu" in Qwen-style architectures)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        SwiGLU Forward Pass process.
        Input x: [Batch, Seq_Len, Hidden_Size]
        """
        # Logging input vector snapshot
        print("🔍 [Yae Element Look] -> Input Vector (First 4 elements):")
        print(f"   {x[0, 0].detach().tolist()}")
        print("-" * 40)

        # 1. [Gate Pathway + Activation]
        gate_out = self.act_fn(self.gate_proj(x))

        # 2. [Parallel Expansion Pathway]
        up_out = self.up_proj(x)

        # 2.5 [SwiGLU multiplication]
        expanded_state = gate_out * up_out

        # 📋 Logging expanded state snapshot
        print(
            "📢 [Yae Element Look] -> Expanded State (Dimension exploded to intermediate_size):"
        )
        formatted_elements = [
            round(num, 4) for num in expanded_state[0, 0].detach().tolist()
        ]
        print(f"   {formatted_elements}")
        print("-" * 40)

        # 3. [Projection back to hidden size]
        output = self.down_proj(expanded_state)

        # Logging final output
        print(
            "🚀 [Yae Element Look] -> Final Output Vector (Compressed back to hidden_size):"
        )
        print(f"   {output[0, 0].detach().tolist()}")
        print("=" * 60)

        return output


if __name__ == "__main__":
    import torch

    class MockConfig:
        hidden_size = 4
        intermediate_size = 12  # Expanded 3x
        hidden_act = "silu"

    torch.manual_seed(42)
    sample_input = torch.randn(1, 2, 4)

    print("=== 📥 1. Starting Input ===")
    print(f"Shape: {list(sample_input.shape)}")
    print("-" * 50)

    # Instantiate YaeMLP
    mlp = YaeMLP(MockConfig())

    print("=== ⚙️ 2. Processing Forward Pass ===")
    final_output = mlp(sample_input)
    print("-" * 50)

    print("=== 🚀 3. Final Output ===")
    print(f"Shape after down_proj: {list(final_output.shape)}")
