import json
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer

from src.encoder import InputHelper
from src.prompt import INS2


class DataCollatorForCSC:
    def __init__(self, tokenizer: PreTrainedTokenizer, input_helper: InputHelper, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.input_helper = input_helper
        self.max_length = max_length
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        self.non_thinking_prefix = "<think>\n\n</think>\n\n"

    def _format_prefix(self, instruction: str) -> str:
        return f"{self.im_start}system\n{instruction}{self.im_end}\n{self.im_start}user\n"

    def _format_prompt(self, instruction: str, src_text: str) -> str:
        return (
            f"{self._format_prefix(instruction)}{src_text}{self.im_end}\n"
            f"{self.im_start}assistant\n{self.non_thinking_prefix}"
        )

    def _format_answer(self, tgt_text) -> str:
        if isinstance(tgt_text, (dict, list)):
            tgt_text = json.dumps(tgt_text, ensure_ascii=False)
        eos_token = self.tokenizer.eos_token or ""
        return f"{tgt_text}{self.im_end}{eos_token}"

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = [INS2 for _ in examples]
        src_texts = [ex["src"] for ex in examples]
        prefix_texts = [self._format_prefix(prompt) for prompt in prompts]
        prompt_texts = [self._format_prompt(prompt, src) for prompt, src in zip(prompts, src_texts)]
        tgt_texts = [self._format_answer(ex["tgt"]) for ex in examples]
        full_texts = [prompt + tgt for prompt, tgt in zip(prompt_texts, tgt_texts)]

        tokenized = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        batch_size, seq_len = input_ids.shape

        labels = input_ids.clone()
        for i in range(batch_size):
            prompt_len = len(self.tokenizer(prompt_texts[i], add_special_tokens=False).input_ids)
            has_bos = (
                self.tokenizer.bos_token_id is not None
                and input_ids[i, 0].item() == self.tokenizer.bos_token_id
            )
            bos_offset = 1 if has_bos else 0
            tgt_start = min(bos_offset + prompt_len, seq_len)
            labels[i, :tgt_start] = -100

        # Mask actual padding positions only. Do not mask by pad_token_id because
        # Qwen commonly uses the same id for pad and eos, and eos must be learned.
        labels = labels.masked_fill(attention_mask == 0, -100)

        pinyins = torch.zeros((batch_size, seq_len, 6), dtype=torch.long, device=input_ids.device)
        images = torch.zeros((batch_size, seq_len, 32, 32), dtype=torch.float32, device=input_ids.device)

        for i in range(batch_size):
            prefix_len = len(self.tokenizer(prefix_texts[i], add_special_tokens=False).input_ids)
            src_len = len(self.tokenizer(src_texts[i], add_special_tokens=False).input_ids)
            has_bos = (
                self.tokenizer.bos_token_id is not None
                and input_ids[i, 0].item() == self.tokenizer.bos_token_id
            )
            bos_offset = 1 if has_bos else 0
            src_start = bos_offset + prefix_len
            src_end = min(src_start + src_len, seq_len)
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
            "labels": labels,
        }
