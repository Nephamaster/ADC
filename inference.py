import argparse
import json
import multiprocessing as mp
import os
import tempfile
from typing import Any, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.prompt import INS2


@torch.no_grad()
def inference_csc(
    model,
    tokenizer,
    input_helper,
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


@torch.no_grad()
def inference_csc_batch(
    model,
    tokenizer,
    input_helper,
    instruction: str,
    src_texts: Sequence[str],
    max_length: int = 1024,
    max_new_tokens: int = 1024,
    device: str = "cuda",
    amp: str = "none",
) -> List[str]:
    if not src_texts:
        return []

    full_texts = [instruction + src_text for src_text in src_texts]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    batch_size, seq_len = input_ids.shape
    prompt_len = len(tokenizer(instruction, add_special_tokens=False).input_ids)
    src_lens = [len(tokenizer(src_text, add_special_tokens=False).input_ids) for src_text in src_texts]

    pinyins = torch.zeros((batch_size, seq_len, 6), dtype=torch.long, device=device)
    images = torch.zeros((batch_size, seq_len, 32, 32), dtype=torch.float32, device=device)

    for i in range(batch_size):
        src_len = src_lens[i]
        has_bos = (tokenizer.bos_token_id is not None and input_ids[i, 0].item() == tokenizer.bos_token_id)
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

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if device.startswith("cuda") and amp in dtype_map:
        with torch.autocast(device_type="cuda", dtype=dtype_map[amp]):
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                phonetic_features=pinyins,
                glyph_features=images,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    else:
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            phonetic_features=pinyins,
            glyph_features=images,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = input_ids.shape[1]
    return [
        tokenizer.decode(generated_ids[i][input_len:], skip_special_tokens=True).strip()
        for i in range(batch_size)
    ]


CSC_INPUT_PREFIX = "*待纠错句子*：\n"
CSC_INPUT_SUFFIX = "\n*正确句子*：\n"


def format_csc_input(original_text: str) -> str:
    return f"{CSC_INPUT_PREFIX}{original_text}{CSC_INPUT_SUFFIX}"


def normalize_vllm_response(raw_text: str, fallback: str) -> str:
    response = raw_text.strip()
    answer_with_think = response.split("</think>")
    # if len(answer_with_think) < 2:
    #     return fallback
    pure_answer = answer_with_think[-1].strip()
    pure_answer = pure_answer.replace("\n", "")
    pure_answer = pure_answer.replace("*待纠错句子", "")
    pure_answer = pure_answer.replace("*正确句子", "")
    pure_answer = pure_answer.replace("*：,", "")
    return pure_answer.strip() or fallback


def run_csc_mode(
    args: argparse.Namespace,
    data: List[str],
) -> List[str]:
    from src.encoder import InputHelper
    from src.modeling_qwen3 import Qwen3ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3ForCausalLM.from_pretrained(
        args.model,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    ).eval()
    input_helper = InputHelper(
        tokenizer,
        cache_dir=args.cache or os.path.join(model.config.name_or_path, "cache"),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preds = []
    for term in tqdm(data, desc="Correcting...", ncols=100):
        src_text = format_csc_input(term)
        pred = inference_csc(
            model=model,
            tokenizer=tokenizer,
            input_helper=input_helper,
            instruction=INS2,
            src_text=src_text,
            max_length=args.max_length,
            device=device
        )
        preds.append(pred.replace('\n','').strip())
    return preds


def build_csc_prompt(tokenizer: AutoTokenizer, text: str) -> str:
    input_text = format_csc_input(text)
    messages = [
        {"role": "system", "content": INS2},
        {"role": "user", "content": input_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_vllm_mode(
    args: argparse.Namespace,
    data: List[str],
    tokenizer: AutoTokenizer,
) -> List[str]:
    from vllm import LLM, SamplingParams
    prompts = [build_csc_prompt(tokenizer, text) for text in data]
    print("Initializing vLLM engine...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    preds: List[str] = []
    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        pure_answer = normalize_vllm_response(raw_text, data[i])
        preds.append(pure_answer)
    return preds


def split_dataset_groups(data: Any, data_file_name:str) -> List[Tuple[str, List[str]]]:
    if isinstance(data, dict):
        return [(name, list(samples)) for name, samples in data.items()]
    if isinstance(data, list):
        return [(data_file_name, list(data))]
    raise ValueError("Unsupported dataset format. Expected list or dict[str, list].")


def write_dataset_predictions(file_path: str, preds: List[str]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(pred + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csc", action="store_true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        if '.json' in args.dataset:
            data = json.load(f)
        elif '.txt' in args.dataset:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line:
                    data.append(line.split('\t')[0])
    data_file_name = args.dataset.split('/')[-1].replace('.jsonl','')
    dataset_groups = split_dataset_groups(data, data_file_name)

    os.makedirs(args.output, exist_ok=True)

    if args.csc:
        print('run csc mode')
        for dataset_name, samples in dataset_groups:
            print(f'Predicting {dataset_name}')
            preds = run_csc_mode(args, samples)
            file_path = os.path.join(args.output, f"{dataset_name}.txt")
            write_dataset_predictions(file_path, preds)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        for dataset_name, samples in dataset_groups:
            preds = run_vllm_mode(args, samples, tokenizer)
            file_path = os.path.join(args.output, f"SFT_{dataset_name}.txt")
            write_dataset_predictions(file_path, preds)
