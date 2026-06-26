import torch
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniVisionEncoderConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniToken2WavConfig,
    Qwen2_5OmniConfig,
    Qwen2_5OmniForConditionalGeneration,
)


text_config = Qwen2_5OmniTextConfig(
    vocab_size=10000,
    hidden_size=1024,
    intermediate_size=2816,
    num_hidden_layers=20,
    num_attention_heads=16,
    num_key_value_heads=16,
)

audio_config = Qwen2_5OmniAudioEncoderConfig(
    d_model=128,
    encoder_layers=2,
    encoder_attention_heads=4,
    encoder_ffn_dim=512,
    num_mel_bins=128,
    output_dim=256,
)

vision_config = Qwen2_5OmniVisionEncoderConfig(
    hidden_size=256,
    depth=2,
    num_heads=4,
    intermediate_size=512,
    out_hidden_size=256,
    fullatt_block_indexes=(1,),
)

thinker_config = Qwen2_5OmniThinkerConfig(
    audio_config=audio_config, vision_config=vision_config, text_config=text_config
)

talker_config = Qwen2_5OmniTalkerConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
    num_key_value_heads=2,
    vocab_size=8192,
)

token2wav_config = Qwen2_5OmniToken2WavConfig()


config = Qwen2_5OmniConfig(
    thinker_config=thinker_config,
    talker_config=talker_config,
    token2wav_config=token2wav_config,
    enable_audio_output=True,
)

model = Qwen2_5OmniForConditionalGeneration(config)

total_params = sum(p.numel() for p in model.parameters())
print("=" * 50)
print(f"📦 Total params: {total_params:,}")
print("=" * 50)
print(model)
