import json
import os

def process(input_path:str, output_path:str):
    with open(input_path, 'r', encoding='utf-8') as f:
        if '.txt' in input_path:
            data = f.readlines()
        elif '.tsv' in input_path:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line:
                    line = line.split('\t')
                    data.append(line[-2]+'\t'+line[-1])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in data:
                line = line.replace(' ','').replace('\n','')
                f.write(line+'\n')


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


def combine(dir:str=None,dataset:str=None):
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
    elif dir is not None:
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

# process(
#     'cscd-ns/test.tsv',
#     'CSCD-NS.txt'
# )

# process_pair(
#     'SIGHAN/test15_error.txt',
#     'SIGHAN/test15_correct.txt',
#     'SIGHAN15.txt'
# )

combine(dataset='LEMON')