import torch
from DistillMiniQwen.model  import Qwen3Attention , Qwen3Config , Qwen3MoeAttention , Qwen3MoEModel
from DistillMiniQwen.util import count_parameters
# ==== Create a dummy config ====
config = Qwen3Config(
            n_embed=512,           # embedding dim (medium size)
            n_head=8,              # 8 attention heads
            n_kv_heads=8,          # same as n_head for normal attention
            n_layer=8,             # 8 transformer blocks
            n_mlp=2048,            # feedforward hidden dim (4× n_embed)
            rope_theta=10000.0,    # rotary position encoding parameter
            rms_norm_eps=1e-5,     # small epsilon for RMSNorm stability
            vocab_size=32000,      # depends on your BPE tokenizer
            tie_word_embeddings=True,
)

# ==== Instantiate the attention module ====
#attn = Qwen3Attention(config)
# attn = Qwen3MoeAttention(config)
# ==== Dummy input ====
B, T, C = 2, 8, config.n_embed  # (batch_size, seq_len, embed_dim)
x = torch.randn(B, T, C)

# ==== Forward pass ====
#with torch.no_grad():
#    y = attn(x)

# ==== Print shapes and check ====
#print(f"Input shape:  {x.shape}")    # [2, 8, 128]
# print(f"Output shape: {y.shape}")    # [2, 8, 128] expected


#  test the model 
model = Qwen3MoEModel(config=config)
total_parms , _ = count_parameters(model)
print(f"num of parms us {total_parms}")