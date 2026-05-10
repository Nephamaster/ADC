import json
import os
import random

def process(input_path:str, output_path:str):
    with open(input_path, 'r', encoding='utf-8') as f:
        if '.txt' in input_path:
            data = f.readlines()
        elif '.tsv' in input_path or 'csv' in input_path:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line:
                    line = line.split('\t')
                    data.append(line[-2]+'\t'+line[-1])
        elif '.jsonl' in input_path:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line:
                    term = json.loads(line)
                    data.append(term['source']+'\t'+term['target'])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in data:
                line = line.replace(' ','').replace('\n','')
                f.write(line+'\n')


def process_train(input_path:list[str], output_path:str):
    random.seed(42)
    data = []
    for path in input_path:
        with open(path, 'r', encoding='utf-8') as f:
            if '.txt' in path or '.tsv' in path or '.csv' in path:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        line = line.split('\t')
                        data.append({'src':line[-2],'tgt':line[-1]})
            elif '.jsonl' in path:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        term = json.loads(line)
                        data.append({'src':term['source'],'tgt':term['target']})
    random.shuffle(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')


def process_pair(err_path:str, corr_path:str, output_path:str):
    with open(err_path, 'r', encoding='utf-8') as f:
        err = f.readlines()
    with open(corr_path, 'r', encoding='utf-8') as f:
        corr = f.readlines()
    with open(output_path, 'w', encoding='utf-8') as f:
        for e, c in zip(err, corr):
            e = e.strip()
            c = c.strip()
            f.write(e+'\t'+c+'\n')


def combine(dir:str,dataset:str=None):
    data = {}
    if dataset is not None:
        for fname in os.listdir(dir):
            if dataset in fname:
                lines = []
                with open(fname,'r',encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            lines.append(line.split('\t')[0])
                    data[fname.replace('.txt','')] = lines
        with open(f'{dataset}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        for fname in os.listdir(dir):
            if '.txt' in fname:
                lines = []
                with open(fname,'r',encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            lines.append(line.split('\t')[0])
                    data[fname.replace('.txt','')] = lines
        with open('CSC_test.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def combine_simple(input_path:list[str],output_path:str):
    data = []
    for path in input_path:
        with open(path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    term = json.loads(line)
                    data.append(term)
    
    random.seed(123)
    random.shuffle(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')

# process(
#     'wang271k/data.jsonl',
#     'Wang271K.txt'
# )

# process_train(
#     [
#         'ecspell/law_train.csv',
#         'ecspell/med_train.csv',
#         'ecspell/odw_train.csv',
#         'cscd-ns/train.tsv',
#         'cscd-ns/dev.tsv',
#         'sighan/SIGHAN-train.txt',
#         'wang271k/data.jsonl'
#     ],
#     'train/csc_train.jsonl'
# )

# process_pair(
#     'sighan/train13_error.txt',
#     'sighan/train13_correct.txt',
#     'SIGHAN13-train.txt'
# )

# combine('.', dataset='rSIGHAN')

combine_simple(
    [
        'train/34m_confuse_gen.jsonl',
        'train/csc_train.jsonl'
    ],
    'train/34m_mix.jsonl'
)