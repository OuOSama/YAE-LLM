# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3 model."""

from collections.abc import Callable

import torch

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.models.gemma.modeling_gemma import GemmaMLP
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .yae_config import YaeConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"


class Qwen3RMSNorm(Qwen2RMSNorm):
    pass


class Qwen3MLP(GemmaMLP):
    pass


class Qwen3RotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class Qwen3Attention(LlamaAttention):
    def __init__(self, config: YaeConfig, layer_idx: int):
        self.layer_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        )
        super().__init__(config, layer_idx)
        self.q_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    def forward(
        self,
        **super_kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        return super().forward(**super_kwargs)


class Qwen3ForSequenceClassification(Qwen2ForSequenceClassification):
    pass


class Qwen3ForTokenClassification(Qwen2ForTokenClassification):
    pass


class Qwen3ForQuestionAnswering(Qwen2ForQuestionAnswering):
    pass
