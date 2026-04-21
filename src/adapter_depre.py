import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from src.encoder import PhoneticEncoder, GlyphEncoder
from transformers.activations import ACT2FN
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3Config


class MLPAdapter(nn.Module):
    def __init__(self,
                 mmodal_dim=56+6,   # glyph + pinyin
                 model_dim=4096,    # llm
                 proj_dim=512,
                 gate_type='residual'):
        super(MLPAdapter, self).__init__()
        self.gate_type = gate_type
        self.fusion = nn.Sequential(
            nn.Linear(mmodal_dim, proj_dim),
            nn.ReLU(),
            # nn.Dropout(0.15),
            nn.Linear(proj_dim, model_dim),
            nn.ReLU(),
            # nn.Dropout(0.15)
        )
        # if gate_type == 'gate_res':
        self.gate = nn.Linear(model_dim, model_dim) # 输出逐通道缩放因子
    
    def forward(self, h_llm, h_phon, h_glyph):
        # h_llm: [B, L, d_model]
        # h_phon, h_glyp: [B, L, mmodal_dim] from NamBert Encoder
        h_fused = torch.cat([h_phon, h_glyph], dim=-1)
        A = self.fusion(h_fused)
        scale = torch.sigmoid(self.gate(A))
        if self.gate_type == 'residual':
            return h_llm + scale*A
        elif self.gate_type == 'scale':
            return h_llm * (scale+1)
    
    @staticmethod
    def from_pretrained(adapter_path, mmodal_dim, model_dim, gate_type):
        adapter = MLPAdapter(mmodal_dim=mmodal_dim, model_dim=model_dim, gate_type=gate_type)
        if 'adapter' not in adapter_path:
            return adapter
        state_dict = torch.load(adapter_path)
        adapter.load_state_dict(state_dict)
        return adapter


class AttnAdapter(nn.Module):
    def __init__(self,
                 config:Qwen3Config,
                 mmodal_dim=56+6,   # glyph + pinyin
                 intermediate_size=512,
                 gate_type='residual',
                 dropout=0.1,
                 ):
        super(AttnAdapter, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.num_attention_heads * self.attention_head_size

        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = nn.Linear(config.hidden_size, 4*config.hidden_size)
        self.mlp_down = nn.Linear(4*config.hidden_size, config.hidden_size)

        self.mmodal_dim = mmodal_dim
        self.mmodal_fusion = nn.Sequential(
            nn.Linear(mmodal_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, config.hidden_size),
            nn.Dropout(dropout),
        )

        self.gate = nn.Linear(config.hidden_size, config.hidden_size)
        self.gate_type = gate_type
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_llm, h_phon, h_glyph):
        # h_llm: [B, L, d_model]
        # h_phon, h_glyp: [B, L, mmodal_dim] from NamBert Encoder
        residual = h_llm
        h_llm_norm = self.input_layernorm(h_llm)
        batch_size, seq_len, _ = h_llm_norm.shape
        h_mmodal = self.mmodal_fusion(torch.cat([h_phon, h_glyph], dim=-1))
        h_mmodal = self.input_layernorm(h_mmodal)

        q = self.q_proj(h_llm_norm).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        k = self.k_proj(h_mmodal).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        v = self.v_proj(h_mmodal).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attention_head_size)
        # attn_scores = torch.einsum('bhld,bhld->bhl', Q, K).unsqueeze(-1)  # [B,H,L,1]
        # attn_scores = attn_scores / math.sqrt(self.attention_head_size)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        attn_output = residual + attn_output
        residual = attn_output
        attn_output_norm = self.post_attention_layernorm(attn_output)
        attn_output_norm = self.mlp_down(self.mlp(attn_output_norm))
        attn_output_norm = residual + attn_output_norm

        scale = torch.sigmoid(self.gate(attn_output_norm))
        if self.gate_type == 'residual':
            h_gate = h_llm + self.dropout(scale*attn_output_norm)
        elif self.gate_type == 'scale':
            h_gate = h_llm * (scale+1)
        else:
            h_gate = h_llm
        
        return h_gate

    @staticmethod
    def from_pretrained(adapter_path, config, mmodal_dim=62, intermediate_size=512, gate_type='residual', dropout=0.1):
        adapter = AttnAdapter(
            config=config, 
            mmodal_dim=mmodal_dim,
            intermediate_size=intermediate_size, 
            gate_type=gate_type,
            dropout=dropout
        )
        if 'adapter' not in adapter_path:
            return adapter
        state_dict = torch.load(adapter_path)
        adapter.load_state_dict(state_dict)
        return adapter