import json
import torch

from typing import Dict, List
from transformers import PreTrainedTokenizer

from src.encoder import InputHelper
from src.prompt import INS, INS2



class DataCollatorForCSC:
    def __init__(self, tokenizer: PreTrainedTokenizer, input_helper: InputHelper, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.input_helper = input_helper
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = [INS2 for ex in examples]
        # prompts = INS
        src_texts = [ex["input"] for ex in examples]
        if isinstance(examples[0]["output"],(dict,list)):
            tgt_texts = [json.dumps(ex["output"], ensure_ascii=False) + self.tokenizer.eos_token for ex in examples]
        else:
            tgt_texts = [ex["output"] + self.tokenizer.eos_token for ex in examples]
        full_texts = [p + s + t for p, s, t in zip(prompts, src_texts, tgt_texts)]
        prompt_len = len(self.tokenizer(prompts[0], add_special_tokens=False)["input_ids"])

        tokenized = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        batch_size, seq_len = input_ids.shape
        
        # 仅 tgt 部分计算损失
        labels = input_ids.clone()
        for i in range(batch_size):
            prompt_len = len(self.tokenizer(prompts[i], add_special_tokens=False).input_ids)
            src_len = len(self.tokenizer(src_texts[i], add_special_tokens=False).input_ids)
            has_bos = (self.tokenizer.bos_token_id is not None 
                    and input_ids[i, 0].item() == self.tokenizer.bos_token_id)
            bos_offset = 1 if has_bos else 0
            tgt_start = bos_offset + prompt_len + src_len
            tgt_start = min(tgt_start, seq_len)  # 安全边界
            labels[i, :tgt_start] = -100
        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
        
        # 与 input_ids 同形状
        pinyins = torch.zeros((batch_size, seq_len, 6), dtype=torch.long, device=input_ids.device)
        images = torch.zeros((batch_size, seq_len, 32, 32), dtype=torch.float32, device=input_ids.device)
        
        for i in range(batch_size):
            prompt_len = len(self.tokenizer(prompts[i], add_special_tokens=False).input_ids)
            src_len = len(self.tokenizer(src_texts[i], add_special_tokens=False).input_ids)
            has_bos = (self.tokenizer.bos_token_id is not None 
                    and input_ids[i, 0].item() == self.tokenizer.bos_token_id)
            bos_offset = 1 if has_bos else 0
            src_start = bos_offset + prompt_len
            src_end = min(bos_offset + prompt_len + src_len, seq_len)
            if src_start >= src_end:
                continue
            src_token_ids = input_ids[i, src_start:src_end]
            pinyin = self.input_helper.convert_tokens_to_pinyin_embeddings(src_token_ids)
            pinyins[i, src_start:src_end, :] = pinyin
            image = self.input_helper.convert_tokens_to_images(src_token_ids, None)
            images[i, src_start:src_end, :, :] = image

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "phonetic_features": pinyins,
            "glyph_features": images,
            "labels": labels
        }