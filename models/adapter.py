import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Tuple

from models.encoder import PhoneticEncoder, GlyphEncoder
from models.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv, Qwen3Attention, Qwen3RMSNorm
from models.configuration_qwen3 import Qwen3Config

from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class CrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        multimodal_feature: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        # cache_position: Optional[torch.LongTensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(multimodal_feature).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(multimodal_feature).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Normalize mask for both eager and SDPA attention backends.
        # Expected output is additive mask in shape [bsz, 1, q_len, k_len]:
        #   0 for keep, large negative for masked.
        if attention_mask is not None:
            q_len = query_states.shape[-2]
            k_len = key_states.shape[-2]
            mask_dtype = query_states.dtype
            neg_value = torch.finfo(mask_dtype).min
            if attention_mask.dim() == 2:
                # [bsz, k_len], usually 1/0 or bool (1/True = valid token).
                key_valid = attention_mask.to(device=query_states.device).bool()
                key_valid = key_valid[:, None, None, :]  # [bsz, 1, 1, k_len]
                attention_mask = (~key_valid).to(dtype=mask_dtype) * neg_value
            elif attention_mask.dim() == 4:
                attention_mask = attention_mask.to(device=query_states.device)
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.logical_not().to(dtype=mask_dtype) * neg_value
                else:
                    attention_mask = attention_mask.to(dtype=mask_dtype)
            else:
                raise ValueError(f"Unsupported attention_mask dim in CrossAttention: {attention_mask.dim()}")
            attention_mask = attention_mask[:, :, :q_len, :k_len]

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
            scaling=self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


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

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.hidden_size,
        #     nhead=num_heads,
        #     dim_feedforward=self.hidden_size * 2,
        #     dropout=dropout,
        #     activation=ACT2FN[config.hidden_act],
        #     batch_first=True,
        #     norm_first=True,
        # )
        # self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.cross_attention = CrossAttention(config=config)

        self.gate_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.gate_norm = Qwen3RMSNorm(self.hidden_size * 2, eps=config.rms_norm_eps)
        self.norm1 = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        phonetic_features: Optional[torch.Tensor] = None,
        glyph_features: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
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
        batch_size, seq_len = hidden_states.shape[:2]
        if seq_len == 1:
            return hidden_states
        residual = hidden_states
        # if phonetic_features is not None and phonetic_features.shape[1] != seq_len:
        #     diff = seq_len - phonetic_features.shape[1]
        #     if diff > 0:
        #         phonetic_features = F.pad(phonetic_features, (0, 0, 0, diff))
        #     else:
        #         phonetic_features = phonetic_features[:, :seq_len, :]
        # if glyph_features is not None and glyph_features.shape[1] != seq_len:
        #     diff = seq_len - glyph_features.shape[1]
        #     if diff > 0:
        #         glyph_features = F.pad(glyph_features, (0, 0, 0, 0, 0, diff))
        #     else:
        #         glyph_features = glyph_features[:, :seq_len, :, :]
        
        phonetic_emb = self.phonetic_encoder(phonetic_features)  # [batch, seq_len, hidden_size]
        glyph_emb = self.glyph_encoder(glyph_features)  # [batch, seq_len, hidden_size]
        
        multimodal_feature = torch.cat([phonetic_emb, glyph_emb], dim=-1)
        multimodal_feature = self.modal_fusion(multimodal_feature)
        
        # 3. Transformer增强上下文
        # 转换attention_mask为Transformer格式（True=padding）
        # transformer_mask = None
        # if attention_mask is not None:
        #     # 确保是 2-D
        #     if attention_mask.dim() > 2:
        #         # 如果维度不对，尝试压缩
        #         attention_mask = attention_mask.view(batch_size, seq_len)
        #     if attention_mask.dtype == torch.bool:
        #         transformer_mask = ~attention_mask
        #     else:
        #         transformer_mask = (attention_mask == 0).bool()
            
        # multimodal_feat = self.fusion_transformer(
        #     multimodal_feat, 
        #     src_key_padding_mask=transformer_mask
        # )
        
        hidden_states_norm = self.norm1(hidden_states)
        attn_output, _ = self.cross_attention(
            hidden_states=hidden_states_norm,
            multimodal_feature=multimodal_feature,
            position_embeddings=position_embeddings,
            attention_mask=padding_mask
        )
        attn_output = self.dropout(attn_output) + residual
        
        gate_input = self.gate_norm(torch.cat([residual, attn_output], dim=-1))
        gate_weight = torch.sigmoid(self.gate_proj(gate_input))
        output = gate_weight * attn_output + (1 - gate_weight) * residual
        output = self.norm2(output)
        
        return output
