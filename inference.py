import os
import torch
import argparse
import json
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from models.modeling_qwen3 import Qwen3ForCausalLM
from models.encoder import InputHelper


@torch.no_grad()
def inference_csc(
    model:Qwen3ForCausalLM,
    tokenizer:AutoTokenizer,
    input_helper:InputHelper,
    instruction: str,
    src_text: str,
    max_length: int = 2048,
    device: str = "cuda"
): 
    full_texts = instruction + src_text
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    batch_size, seq_len = input_ids.shape
    prompt_len = len(tokenizer(instruction, add_special_tokens=False).input_ids)
    # 与 input_ids 同形状
    pinyins = torch.zeros((batch_size, seq_len, 6), dtype=torch.long, device=device)
    images = torch.zeros((batch_size, seq_len, 32, 32), dtype=torch.float32, device=device)
    
    for i in range(batch_size):
        src_len = len(tokenizer(src_text, add_special_tokens=False).input_ids)
        has_bos = (tokenizer.bos_token_id is not None 
                and input_ids[i, 0].item() == tokenizer.bos_token_id)
        bos_offset = 1 if has_bos else 0
        src_start = bos_offset + prompt_len
        src_end = min(bos_offset + prompt_len + src_len, seq_len)
        if src_start >= src_end:
            continue
        src_token_ids = input_ids[i, src_start:src_end]
        pinyin = input_helper.convert_tokens_to_pinyin_embeddings(src_token_ids)
        pinyins[i, src_start:src_end, :] = pinyin.to(device)
        image = input_helper.convert_tokens_to_images(src_token_ids, None)
        images[i, src_start:src_end, :, :] = image.to(device)
    
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phonetic_features=pinyins,
        glyph_features=images,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ===== 使用示例 =====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)  # CSC任务推荐0.0（确定性输出）
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    ).eval()
    input_helper = InputHelper(tokenizer, cache_dir=args.cache_dir or os.path.join(model.config.name_or_path, "cache"))
    
    data = []
    with open(args.dataset_path,'r',encoding='utf-8') as f:
        # for line in tqdm(f.readlines(),desc='Reading data...',ncols=100):
        #     data.append(json.loads(line))
        data = json.load(f)
    
    INST = """请检测待纠错句子中的中文拼写错误。

**格式要求**
- 输出句子中每处错误的位置(索引从0开始)，错字和正确字
- 每项错误纠正间用换行符隔开
- 如果没有错误，则输出"无错误"
- 除此之外不要输出任何其他内容

---
示例 1：
*待纠错句子*：
今天天汽真不搓。
*纠错结果*：
3, 汽, 气
6, 搓, 错

示例 2：
*待纠错句子*：
我要吃早惨。
*纠错结果*：
4, 惨, 餐

示例 3：
*待纠错句子*：
今年是我的本命年。
*纠错结果*：
无错误
---
现在请对以下句子进行纠错：
"""
    corrects = []
    with open('pred_ADC_attn_28.jsonl', 'w', encoding='utf-8') as f:
        for term in tqdm(data, desc='Correcting...',ncols=100,total=len(data)):
            corrected = inference_csc(
                model=model,
                tokenizer=tokenizer,
                input_helper=input_helper,
                instruction=INST,
                src_text="*待纠错句子*：\n"+term['original_text']+"\n*纠错结果*：\n",
                max_length=args.max_length,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # line = json.loads(corrected)
            f.write(json.dumps(corrected, ensure_ascii=False) + '\n')
            f.flush()
            # corrects.append(corrected)
    
    # with open('pred.jsonl','a',encoding='utf-8') as f:
    #     # f.write('\n'.join(corrects))
    #     for item in corrects:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')