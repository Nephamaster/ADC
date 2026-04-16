import json

def count_err(input_path:str):
    total, error = 0, 0
    with open(input_path, 'r', encoding='utf-8') as f:
        if 'jsonl' in input_path:
            for line in f.readlines():
                line = line.strip()
                if line:
                    term = json.loads(line)
                    src = term['input'].replace('\n*纠错结果*：','').replace('*待纠错句子*：\n','').strip()
                    tgt = term['output'].strip()
                    if src != tgt:
                        error += 1
                    total += 1
        elif 'txt' in input_path:
            for line in f.readlines():
                line = line.strip()
                if line:
                    line = line.split('\t')
                    src, tgt = line[0].strip(), line[1].strip()
                    if src != tgt:
                        error += 1
                    total += 1
    print('Error Ratio:', error*1.0/total)

count_err('../../../LlamaFactory/data/csc_mix.jsonl')
# count_err('CSCD-NS-train.txt')