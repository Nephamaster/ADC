import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from models.encoder import PhoneticEncoder, GlyphEncoder
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


class CSCAdapter(nn.Module):
    """
    插入位置：Qwen3DecoderLayer的Attention之后、MLP之前
    """
    def __init__(self, config, num_heads:int=4, dropout:float=0.1):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = num_heads

        self.phonetic_encoder = PhoneticEncoder(self.hidden_size, dropout)
        self.glyph_encoder = GlyphEncoder(self.hidden_size, font_size=32, dropout=dropout)
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size, bias=False),
            Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps),
            ACT2FN[config.hidden_act],  # SwiGLU
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            dim_feedforward=self.hidden_size * 2,
            dropout=dropout,
            activation=ACT2FN[config.hidden_act],
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 4. 交叉注意力（Q=隐藏状态，KV=多模态特征）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 5. 门控融合机制
        self.gate_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.gate_norm = Qwen3RMSNorm(self.hidden_size * 2, eps=config.rms_norm_eps)
        
        # 6. 归一化（RMSNorm与Qwen3一致）
        self.norm1 = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        phonetic_features: Optional[torch.Tensor] = None,
        glyph_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]，来自Attention的输出
            phonetic_features: [batch, seq_len, 6]，拼音特征
            glyph_features: [batch, seq_len, 32, 32]，汉字图像
            attention_mask: [batch, seq_len]，注意力掩码（1=有效，0=padding）
        Returns:
            [batch, seq_len, hidden_size] 融合后的隐藏状态
        """
        residual = hidden_states
        batch_size, seq_len = hidden_states.shape[:2]
        
        # 1. 编码字音字形特征
        phonetic_emb = self.phonetic_encoder(phonetic_features)  # [batch, seq_len, hidden_size]
        glyph_emb = self.glyph_encoder(glyph_features)  # [batch, seq_len, hidden_size]
        
        # 2. 多模态融合
        multimodal_feat = torch.cat([phonetic_emb, glyph_emb], dim=-1)
        multimodal_feat = self.modal_fusion(multimodal_feat)
        
        # 3. Transformer增强上下文
        # 转换attention_mask为Transformer格式（True=padding）
        transformer_mask = None
        if attention_mask is not None:
            # 确保是 2-D
            if attention_mask.dim() > 2:
                # 如果维度不对，尝试压缩
                attention_mask = attention_mask.view(batch_size, seq_len)
            if attention_mask.dtype == torch.bool:
                transformer_mask = ~attention_mask
            else:
                transformer_mask = (attention_mask == 0).bool()
            
        multimodal_feat = self.fusion_transformer(
            multimodal_feat, 
            src_key_padding_mask=transformer_mask
        )
        
        # 4. 交叉注意力（Q=hidden_states, K=V=multimodal_feat）
        hidden_states_norm = self.norm1(hidden_states)
        attn_output, _ = self.cross_attention(
            query=hidden_states_norm,
            key=multimodal_feat,
            value=multimodal_feat,
            key_padding_mask=transformer_mask,
        )
        attn_output = self.dropout(attn_output) + residual
        
        # 5. 门控融合（自适应平衡原始语义与纠错信息）
        gate_input = self.gate_norm(torch.cat([residual, attn_output], dim=-1))
        gate_weight = torch.sigmoid(self.gate_proj(gate_input))
        output = gate_weight * attn_output + (1 - gate_weight) * residual
        
        # 6. 最终归一化
        output = self.norm2(output)
        
        return output