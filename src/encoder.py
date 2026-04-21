import argparse
import os
import torch
import pickle
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from src.utils import convert_char_to_pinyin, convert_char_to_image, pred_token_process
from typing import List, Optional, Tuple


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InputHelper:
    """
    @NamBert
    """
    def __init__(self, tokenizer, cache_dir=None):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.pinyin_embedding_cache = None
        self._init_pinyin_embedding_cache()
        self.token_images_cache = None
        self._init_token_images_cache()

    def _init_pinyin_embedding_cache(self):
        self.pinyin_embedding_cache = {}
        if os.path.exists(os.path.join(self.cache_dir,'pinyin_embedding_cache.pkl')):
            print('Loading pinyin embedding cache...')
            with open(os.path.join(self.cache_dir,'pinyin_embedding_cache.pkl'), 'rb') as f:
                self.pinyin_embedding_cache = pickle.load(f)
        else:
            for token, id in tqdm(self.tokenizer.get_vocab().items(), desc='Initializing pinyin embeddings cache...', ncols=100):
                self.pinyin_embedding_cache[id] = convert_char_to_pinyin(token, tone=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(os.path.join(self.cache_dir,'pinyin_embedding_cache.pkl'), 'wb') as f:
                pickle.dump(self.pinyin_embedding_cache, f)

    def _init_token_images_cache(self):
        self.token_images_cache = {}
        if os.path.exists(os.path.join(self.cache_dir,'token_image_cache.pkl')):
            print('Loading token images cache...')
            with open(os.path.join(self.cache_dir,'token_image_cache.pkl'), 'rb') as f:
                self.token_images_cache = pickle.load(f)
        else:
            for token, id in tqdm(self.tokenizer.get_vocab().items(), desc='Initializing token images cache...', ncols=100):
                self.token_images_cache[id] = convert_char_to_image(token, 32)
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(os.path.join(self.cache_dir,'token_image_cache.pkl'), 'wb') as f:
                pickle.dump(self.token_images_cache, f)

    def convert_tokens_to_pinyin_embeddings(self, input_ids):
        input_pinyins = []
        for i, input_id in enumerate(input_ids):
            input_pinyins.append(self.pinyin_embedding_cache.get(input_id.item(), torch.LongTensor([0])))
        return pad_sequence(input_pinyins, batch_first=True)

    def convert_tokens_to_images(self, input_ids, characters):
        images = []
        for i, input_id in enumerate(input_ids):
            if input_id == 100:
                if characters and i - 1 > 0 and i - 1 < len(characters):
                    images.append(convert_char_to_image(characters[i - 1], 32))
                    continue
            images.append(self.token_images_cache.get(input_id.item(), torch.zeros(32, 32)))
        return torch.stack(images)


# class InputHelper:
#     """
#     适配CSC-Adapter的输入辅助类
#     @NamBert
#     """
#     def __init__(self, tokenizer, cache_dir: str = None):
#         self.tokenizer = tokenizer
#         self.cache_dir = cache_dir or './csc_cache'
#         self.pinyin_embedding_cache = None
#         self._init_pinyin_embedding_cache()
#         self.token_images_cache = None
#         self._init_token_images_cache()

#     def _init_pinyin_embedding_cache(self):
#         """初始化拼音特征缓存"""
#         cache_path = os.path.join(self.cache_dir, 'pinyin_embedding_cache.pkl')
#         self.pinyin_embedding_cache = {}
        
#         if os.path.exists(cache_path):
#             print('Loading pinyin embedding cache...')
#             with open(cache_path, 'rb') as f:
#                 self.pinyin_embedding_cache = pickle.load(f)
#         else:
#             for token, token_id in tqdm(
#                 self.tokenizer.get_vocab().items(), 
#                 desc='Initializing pinyin embeddings cache...', 
#                 ncols=100
#             ):
#                 # convert_char_to_pinyin 返回6维拼音特征
#                 pinyin_feat = convert_char_to_pinyin(token, tone=True)
#                 self.pinyin_embedding_cache[token_id] = pinyin_feat
            
#             os.makedirs(self.cache_dir, exist_ok=True)
#             with open(cache_path, 'wb') as f:
#                 pickle.dump(self.pinyin_embedding_cache, f)

#     def _init_token_images_cache(self):
#         """初始化汉字图像缓存"""
#         cache_path = os.path.join(self.cache_dir, 'token_image_cache.pkl')
#         self.token_images_cache = {}
        
#         if os.path.exists(cache_path):
#             print('Loading token images cache...')
#             with open(cache_path, 'rb') as f:
#                 self.token_images_cache = pickle.load(f)
#         else:
#             for token, token_id in tqdm(
#                 self.tokenizer.get_vocab().items(), 
#                 desc='Initializing token images cache...', 
#                 ncols=100
#             ):
#                 # convert_char_to_image 返回32x32图像
#                 image = convert_char_to_image(token, 32)
#                 self.token_images_cache[token_id] = image
            
#             os.makedirs(self.cache_dir, exist_ok=True)
#             with open(cache_path, 'wb') as f:
#                 pickle.dump(self.token_images_cache, f)

#     def batch_get_pinyin_features(
#         self, 
#         input_ids: torch.Tensor,
#         device: torch.device = None
#     ) -> torch.Tensor:
#         """
#         获取批量拼音特征
#         Args:
#             input_ids: [batch, seq_len]
#         Returns:
#             [batch, seq_len, 6]
#         """
#         if device is None:
#             device = input_ids.device
            
#         batch_pinyins = []
#         for batch_idx in range(input_ids.shape[0]):
#             batch_pinyins.append([])
#             for token_id in input_ids[batch_idx]:
#                 feat = self.pinyin_embedding_cache.get(token_id.item(), [0] * 6)
#                 if not isinstance(feat, torch.Tensor):
#                     feat = torch.tensor(feat, dtype=torch.float32)
#                 batch_pinyins[-1].append(feat)
        
#         # 填充为统一长度
#         result = []
#         for batch_pinyin in batch_pinyins:
#             result.append(torch.stack(batch_pinyin))
        
#         return torch.stack(result).to(device)

#     def batch_get_glyph_features(
#         self, 
#         input_ids: torch.Tensor,
#         device: torch.device = None
#     ) -> torch.Tensor:
#         """
#         获取批量字形特征（图像）
#         Args:
#             input_ids: [batch, seq_len]
#         Returns:
#             [batch, seq_len, 32, 32]
#         """
#         if device is None:
#             device = input_ids.device
            
#         batch_images = []
#         for batch_idx in range(input_ids.shape[0]):
#             batch_images.append([])
#             for token_id in input_ids[batch_idx]:
#                 img = self.token_images_cache.get(token_id.item(), torch.zeros(32, 32))
#                 if not isinstance(img, torch.Tensor):
#                     img = torch.tensor(img, dtype=torch.float32)
#                 batch_images[-1].append(img)
        
#         # 堆叠为统一形状
#         result = []
#         for batch_img in batch_images:
#             result.append(torch.stack(batch_img))
        
#         return torch.stack(result).to(device)

#     def batch_encode(
#         self, 
#         input_ids: torch.Tensor,
#         device: torch.device = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         同时获取拼音和字形特征
#         Returns:
#             phonetic_features: [batch, seq_len, 6]
#             glyph_features: [batch, seq_len, 32, 32]
#         """
#         phonetic_features = self.batch_get_pinyin_features(input_ids, device)
#         glyph_features = self.batch_get_glyph_features(input_ids, device)
#         return phonetic_features, glyph_features

# class PhoneticEncoder(nn.Module):
#     """
#     @NamBert
#     """
#     def __init__(self):
#         super(PhoneticEncoder, self).__init__()
#         self.pinyin_feature_size = 6
#         self.embedding_layer = nn.Linear(6, 6, bias=True)

#     def forward(self, inputs):
#         fill = self.pinyin_feature_size - inputs.size(1)
#         if fill > 0:
#             inputs = torch.concat([inputs, torch.zeros((len(inputs), fill))], dim=1).long()
#         target_dtype = self.embedding_layer.weight.dtype
#         inputs = self.embedding_layer(inputs.to(target_dtype))
#         return inputs

#     @staticmethod
#     def from_pretrained(pretrained_model_path):
#         phon_embedding = PhoneticEncoder()
#         if 'phonetic' not in pretrained_model_path:
#             return phon_embedding
#         state_dict = torch.load(pretrained_model_path+'/phonetic.bin')
#         phon_embedding.load_state_dict(state_dict)
#         return phon_embedding


# class GlyphEncoder(nn.Module):
#     """
#     @NamBert
#     """
#     def __init__(self, font_size=32):
#         super(GlyphEncoder, self).__init__()
#         self.font_size = font_size
#         self.embeddings = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.15),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.15),
#             nn.Linear(256, 56),
#             nn.Tanh()
#         )

#     def forward(self, images):
#         batch_size, seq_len = images.shape[0], images.shape[1]
#         target_dtype = self.embeddings[0].weight.dtype
#         images = (images.view(batch_size, seq_len, -1) / 255.).to(target_dtype)
#         return self.embeddings(images)

#     @staticmethod
#     def from_pretrained(pretrained_model_path):
#         glyph_embedding = GlyphEncoder()
#         if 'glyph' not in pretrained_model_path:
#             return glyph_embedding
#         state_dict = torch.load(pretrained_model_path+'/glyph.bin')
#         glyph_embedding.load_state_dict(state_dict)
#         return glyph_embedding


# class PhoneticEncoder(nn.Module):
#     """字音编码器"""
#     def __init__(self, hidden_size: int, dropout: float = 0.1):
#         super().__init__()
#         # 拼音三要素（声母23 + 韵母35 + 声调5）
#         self.initial_embed = nn.Embedding(23, 128)
#         self.final_embed = nn.Embedding(35, 128)
#         self.tone_embed = nn.Embedding(5, 64)
        
#         self.phonetic_fusion = nn.Sequential(
#             nn.Linear(320, hidden_size, bias=False),
#             Qwen3RMSNorm(hidden_size),
#             nn.Dropout(dropout)
#         )
    
#     def forward(self, pinyin_ids: torch.Tensor) -> torch.Tensor:
#         # pinyin_ids: [batch, seq_len, 3]
#         init_emb = self.initial_embed(pinyin_ids[:, :, 0])
#         final_emb = self.final_embed(pinyin_ids[:, :, 1])
#         tone_emb = self.tone_embed(pinyin_ids[:, :, 2])
#         phonetic_feat = torch.cat([init_emb, final_emb, tone_emb], dim=-1)
#         return self.phonetic_fusion(phonetic_feat)


# class GlyphEncoder(nn.Module):
#     """字形编码器"""
#     def __init__(self, hidden_size: int, dropout: float = 0.1):
#         super().__init__()
#         # 部首214 + 笔画30 + 四角号码10
#         self.radical_embed = nn.Embedding(214, 256)
#         self.stroke_embed = nn.Embedding(30, 128)
#         self.corner_embed = nn.Embedding(10, 64)
        
#         self.glyph_fusion = nn.Sequential(
#             nn.Linear(448, hidden_size, bias=False),
#             Qwen3RMSNorm(hidden_size),
#             nn.Dropout(dropout)
#         )
    
#     def forward(self, glyph_ids: torch.Tensor) -> torch.Tensor:
#         # glyph_ids: [batch, seq_len, 3]
#         rad_emb = self.radical_embed(glyph_ids[:, :, 0])
#         stroke_emb = self.stroke_embed(glyph_ids[:, :, 1])
#         corner_emb = self.corner_embed(glyph_ids[:, :, 2])
#         glyph_feat = torch.cat([rad_emb, stroke_emb, corner_emb], dim=-1)
#         return self.glyph_fusion(glyph_feat)

class PhoneticEncoder(nn.Module):
    """
    输入: [batch, seq_len, 6] 拼音特征
    输出: [batch, seq_len, hidden_size]
    """
    def __init__(self, hidden_size:int=4096, dropout:float=0.1):
        super().__init__()
        self.pinyin_feature_size = 6
        self.original_embedding = nn.Linear(6, 6, bias=True)
        self.phonetic_projection = nn.Sequential(
            nn.Linear(6, 256, bias=False),
            Qwen3RMSNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_size, bias=False),
            Qwen3RMSNorm(hidden_size),
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, seq_len, 6] 拼音特征
        Returns:
            [batch, seq_len, hidden_size]
        """
        fill = self.pinyin_feature_size - inputs.size(-1)
        if fill > 0:
            inputs = torch.cat([inputs, torch.zeros((*inputs.shape[:2], fill), 
                                                    device=inputs.device)], dim=-1)
        target_dtype = self.original_embedding.weight.dtype
        phonetic_feat = self.original_embedding(inputs.to(target_dtype))
        # 投影到hidden_size
        phonetic_feat = self.phonetic_projection(phonetic_feat)
        return phonetic_feat


class GlyphEncoder(nn.Module):
    """
    输入: [batch, seq_len, 32, 32] 汉字图像
    输出: [batch, seq_len, hidden_size]
    """
    def __init__(self, hidden_size:int=4096, font_size:int=32, dropout:float=0.1):
        super().__init__()
        self.font_size = font_size
        self.original_embeddings = nn.Sequential(
            nn.Linear(font_size**2, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 56),
            nn.Tanh()
        )
        self.glyph_projection = nn.Sequential(
            nn.Linear(56, 256, bias=False),
            Qwen3RMSNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_size, bias=False),
            Qwen3RMSNorm(hidden_size),
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, seq_len, 32, 32] 汉字图像
        Returns:
            [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = images.shape[:2]
        target_dtype = self.original_embeddings[0].weight.dtype
        images_flat = (images.view(batch_size, seq_len, -1) / 255.).to(target_dtype)
        glyph_feat = self.original_embeddings(images_flat)
        glyph_feat = self.glyph_projection(glyph_feat)
        return glyph_feat