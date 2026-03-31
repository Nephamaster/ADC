import os
import torch
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm


MODEL_PATH = "./Qwen3-8B-Adapter/checkpoint-11935" 
TP_SIZE = 1 
GPU_MEM_UTIL = 0.9
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


with open("../MSLLM/data/test.json", 'r', encoding='utf-8') as f:
    csc_data = json.load(f)

print("加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def build_csc_prompt(term):
    """
    构建符合 Qwen Instruct 格式的纠错指令
    """
    input = "*待纠错句子*：\n"+term['original_text']+"\n*纠错结果*：\n"
    messages = [
        {"role": "system", "content": f"{INST}"},
        {"role": "user", "content": f"{input}"}
    ]
    # 使用 tokenizer 应用聊天模板，确保模型能正确理解指令
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

prompts = [build_csc_prompt(term) for term in csc_data]


print("初始化 vLLM 引擎...")
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    dtype="auto",
    tensor_parallel_size=TP_SIZE,
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=4096,
)


sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=4096,
    stop=["<|im_end|>"]
)


print("开始批量推理...")
outputs = llm.generate(prompts, sampling_params)

corrects = []
with open('pred_ADC_sft.jsonl', 'w', encoding='utf-8') as f:
    for output in tqdm(outputs, desc='Correcting...',ncols=100,total=len(outputs)):
        # line = json.loads(corrected)
        answer_with_think = output.outputs[0].text.strip().split('</think>')
        pure_answer = answer_with_think[1] if len(answer_with_think) >= 2 else answer_with_think[0]
        f.write(json.dumps(pure_answer.strip(), ensure_ascii=False) + '\n')
        f.flush()