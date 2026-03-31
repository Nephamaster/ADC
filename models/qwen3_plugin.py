import torch
import torch.nn as nn
import types
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from models.adapter import MLPAdapter, AttnAdapter
from models.encoder import PhoneticEncoder, GlyphEncoder


# def make_forward_with_adapter(original_forward):
#     """Factory function to create a new forward with adapter injection."""
#     def new_forward(self, hidden_states, *args, **kwargs):
#         if hasattr(self, 'adapter') and self.adapter is not None:
#             hidden_states = self.adapter(hidden_states, self.phon_emb, self.glyp_emb)
#         return original_forward(self, hidden_states, *args, **kwargs)
#     return new_forward


def make_forward_with_adapter(original_forward):
    """
    Factory function to create a new forward with adapter injection.
    Safe for: Pure text inference, Multimodal inference, and KV Cache generation.
    """
    def new_forward(self, hidden_states, *args, **kwargs):
        phon_emb = getattr(self, 'phon_emb', None)
        glyp_emb = getattr(self, 'glyp_emb', None)
        should_inject = (
            hasattr(self, 'adapter') and 
            self.adapter is not None and
            phon_emb is not None and 
            glyp_emb is not None
        )
        if should_inject:
            batch_size, h_len, _ = hidden_states.shape
            p_len = phon_emb.shape[1]
            position_ids = kwargs.get('position_ids', None)
            if h_len == p_len:
                hidden_states = self.adapter(hidden_states, phon_emb, glyp_emb)
            elif h_len == 1:
                if position_ids is not None:
                    curr_idx = position_ids[:, -1:] 
                    if curr_idx.max() < p_len:
                        idx_expanded = curr_idx.unsqueeze(-1).expand(-1, -1, phon_emb.shape[-1])
                        p_slice = torch.gather(phon_emb, 1, idx_expanded)
                        
                        g_idx_expanded = curr_idx.unsqueeze(-1).expand(-1, -1, glyp_emb.shape[-1])
                        g_slice = torch.gather(glyp_emb, 1, g_idx_expanded)
                        
                        hidden_states = self.adapter(hidden_states, p_slice, g_slice)
                else:
                    pass
        return original_forward(self, hidden_states, *args, **kwargs)
        
    return new_forward


class Qwen3InnerPlugin(Qwen3ForCausalLM):
    """
    Qwen3ForCausalLM + Multimodal Adapter
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        # self._setup_(kwargs.get("model_path"), **kwargs)
    
    def _setup_(self, model_path, **kwargs):
        self.gate_type = kwargs.get('gate_type', 'residual')
        self.plug_idx = kwargs.get('plug_idx', [-6])
        self.phonetic = PhoneticEncoder().from_pretrained(model_path)
        self.glyph = GlyphEncoder().from_pretrained(model_path)
        device = getattr(self, "device", torch.device("cpu"))
        dtype = getattr(self, "dtype", torch.float32)
        self.phonetic.to(device=device, dtype=dtype)
        self.glyph.to(device=device, dtype=dtype)
        self.phon_size, self.glyph_size = 6, 56
        self._replace_layer_forwards(model_path)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        model = super().from_pretrained(model_path, **kwargs)
        # Convert to InnerPlugin
        # model.__class__ = cls
        model._setup_(model_path, **kwargs)
        return model

    def _replace_layer_forwards(self, model_path):
        """确保 adapter 与所属 layer 的设备和精度完全一致"""
        original_forward = Qwen3DecoderLayer.forward
        new_forward_fn = make_forward_with_adapter(original_forward)
        # 获取主模型的设备和精度作为基准
        target_device = self.device
        target_dtype = self.dtype
        for idx in self.plug_idx:
            layer = self.model.layers[idx]
            layer.forward = types.MethodType(new_forward_fn, layer)
            layer.adapter = AttnAdapter.from_pretrained(
                model_path,
                config=self.config,
                mmodal_dim=self.phon_size + self.glyph_size,
                gate_type=self.gate_type,
            )
            layer.adapter.to(device=target_device, dtype=target_dtype)

    # def _replace_layer_forwards(self, model_path):
    #     """Replace forward method of specified layers once at init."""
    #     original_forward = Qwen3DecoderLayer.forward
    #     new_forward_fn = make_forward_with_adapter(original_forward)
    #     for idx in self.plug_idx:
    #         layer = self.model.layers[idx]
    #         layer.forward = types.MethodType(new_forward_fn, layer)
    #         layer.adapter = MLPAdapter.from_pretrained(model_path,
    #         mmodal_dim=self.phon_size + self.glyph_size,
    #         model_dim=self.config.hidden_size, gate_type=self.gate_type)

    def inject_multimodal_features(self, phon_embed, glyp_emb):
        """Inject multimodal features into plugged layers."""
        for pidx in self.plug_idx:
            layer = self.model.layers[pidx]
            layer.phon_emb = phon_embed
            layer.glyp_emb = glyp_emb

    def clear_multimodal_features(self):
        for pidx in self.plug_idx:
            layer = self.model.layers[pidx]
            if hasattr(layer, 'phon_emb'): del layer.phon_emb
            if hasattr(layer, 'glyp_emb'): del layer.glyp_emb

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        pinyins: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        try:
            if pinyins is not None and images is not None:
                phon_emb:torch.Tensor = self.phonetic(pinyins)
                phon_emb = phon_emb.view(batch_size, -1, self.phon_size)
                glyp_emb:torch.Tensor = self.glyph(images)
                glyp_emb = glyp_emb.view(batch_size, -1, self.glyph_size)
                # assert phon_emb.shape[:2] == (batch_size, seq_len), f"Phon shape mismatch: {phon_emb.shape}"
                # assert glyp_emb.shape[:2] == (batch_size, seq_len), f"Glyph shape mismatch: {glyp_emb.shape}"
                self.inject_multimodal_features(phon_emb, glyp_emb)

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
            return outputs
        finally:
            self.clear_multimodal_features()
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        
        # 透传多模态特征（KV Cache 模式下仅需首次传入 full-seq 特征）
        if "pinyins" in kwargs:
            model_inputs["pinyins"] = kwargs["pinyins"]
        if "images" in kwargs:
            model_inputs["images"] = kwargs["images"]
        
        return model_inputs