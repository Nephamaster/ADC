import os.path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

class VocabularyPruner(object):

    def check(self, old_model_name_or_path, new_model_name_or_path, text):
        # 检查模型裁剪后，生成结果是否一致
        max_length = 10
        # 使用老模型对文本编码
        old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path,trust_remote_code=True)
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path,trust_remote_code=True,use_fast=False)
        old_input_ids = old_tokenizer(text, return_tensors='pt').input_ids
        old_output = old_model.generate(old_input_ids, max_new_tokens=max_length, do_sample=False, num_beams=1)
        old_output_text = old_tokenizer.batch_decode(old_output)
        print('old_output:{}'.format(old_output_text))
        # 使用新模型对文本编码
        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path,trust_remote_code=True)
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_path,trust_remote_code=True,use_fast=False)
        new_input_ids = new_tokenizer(text, return_tensors='pt').input_ids
        new_output = new_model.generate(new_input_ids, max_new_tokens=max_length, do_sample=False, num_beams=1)
        new_output_text = new_tokenizer.batch_decode(new_output)
        print('new_output:{}'.format(new_output_text))
        if old_output_text == new_output_text:
            print('output is same, succeed to prune.')
        else:
            print('output is not same, fail to prune.')

    def check_embedding(self, old_model_name_or_path, new_model_name_or_path, text):
        # 检查模型裁剪后，生成结果是否一致
        # 使用老模型对文本编码
        old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path,trust_remote_code=True)
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path,trust_remote_code=True,use_fast=False)
        old_token = old_tokenizer.tokenize(text)
        print('old_token:{}'.format(old_token))
        old_input_ids = old_tokenizer(text, return_tensors='pt').input_ids
        print('old_input_ids:{}'.format(old_input_ids))
        old_model_input_embed = old_model.get_input_embeddings().weight.data[old_tokenizer(text, return_tensors='pt').input_ids]
        print("old_model_input_embed: ",old_model_input_embed)
        old_model_output_embed = old_model.get_output_embeddings().weight.data[old_tokenizer(text, return_tensors='pt').input_ids]
        print("old_model_output_embed: ", old_model_output_embed)

        # 使用新模型对文本编码
        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path,trust_remote_code=True)
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_path,trust_remote_code=True,use_fast=False)
        new_token = new_tokenizer.tokenize(text)
        print('new_token:{}'.format(new_token))
        new_input_ids = new_tokenizer(text, return_tensors='pt').input_ids
        print('new_input_ids:{}'.format(new_input_ids))
        new_model_input_embed = new_model.get_input_embeddings().weight.data[new_tokenizer(text, return_tensors='pt').input_ids]
        print("new_model_input_embed: ",new_model_input_embed)
        new_model_output_embed = new_model.get_output_embeddings().weight.data[new_tokenizer(text, return_tensors='pt').input_ids]
        print("new_model_output_embed: ",new_model_output_embed)
        print("equal? ", all((old_model_input_embed == new_model_input_embed)[0][0]), all((old_model_output_embed == new_model_output_embed)[0][0]))
    
    def update_embeddings(self, model, new2old_token_id, new_embeds, new_lm_head):
        raise NotImplemented

    def prune(self, model_name_or_path, new_tokenizer_name_or_path, save_path, new_name_or_path=None):
        # 创建输出目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 加载新词表
        new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_name_or_path,trust_remote_code=True,use_fast=False)
        # 加载原词表
        old_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True,use_fast=False)
        special_tokens_to_preserve = {
            'pad_token': old_tokenizer.pad_token,
            'eos_token': old_tokenizer.eos_token, 
            'bos_token': old_tokenizer.bos_token,
            'unk_token': old_tokenizer.unk_token,
        }
        print(f"Special tokens to preserve: {special_tokens_to_preserve}")
        print("new_tokenizer:",len(new_tokenizer)," old_tokenizer: ",len(old_tokenizer))
        # 检查新词表是否为原词表的子集
        old_vocab = old_tokenizer.get_vocab()
        new_vocab = new_tokenizer.get_vocab()
        for token in tqdm(new_vocab.keys()):
            if token not in old_vocab:
                raise Exception('{} not exist'.format(token))
        print('new_tokenizer is subset of old_tokenizer')

        # 获得新词表中每个token_id到原词表的token_id的映射
        new2old_token_id = {}
        for token, token_id in tqdm(new_vocab.items()):
            old_token_id = old_vocab[token]
            # print(token, token_id)
            # if token_id >= 151643:
            #     print('token_id >= 151643')
            #     new2old_token_id[135269+token_id-151643] = old_token_id
            # else:
            new2old_token_id[token_id] = old_token_id

        for i in range(0,267): new2old_token_id[135295+i] = 151669+i
        # 267 = 151936-151669
        for token_name, token_value in special_tokens_to_preserve.items():
            if token_value and token_value in new_vocab:
                new_token_id = new_vocab[token_value]
                old_token_id = old_vocab[token_value]
                # 确保映射一致
                assert new2old_token_id[new_token_id] == old_token_id, \
                    f"{token_name} mapping mismatch!"
                print(f"✅ {token_name} ('{token_value}'): new_id={new_token_id} → old_id={old_token_id}")
            elif token_value:
                print(f"⚠️  Warning: {token_name} ('{token_value}') not in new_vocab!")

        # 加载多语言模型
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype='auto', trust_remote_code=True)
        print("Loaded model config vocab_size:", model.config.vocab_size)
        print("Embedding weight shape:", model.get_input_embeddings().weight.shape)
        # 计算原模型的参数量
        old_params = sum(p.numel() for p in model.parameters())
        print("Total params of original model: %.2fM" % (old_params / 1e6))

        # 对于新词表中的每个token，取出其对应的权重，复制到新模型中
        vocab_size = len(new2old_token_id)
        hidden_size = model.config.hidden_size

        new_embeds = torch.nn.Embedding(vocab_size, hidden_size, dtype=model.dtype)
        new_lm_head = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=False, dtype=model.dtype)
        
        
        # 更新词表权重
        self.update_embeddings(model, new2old_token_id, new_embeds, new_lm_head)

        model.config.__dict__['vocab_size'] = vocab_size
        if new_name_or_path is not None:
            model.config.__dict__['_name_or_path'] = new_name_or_path

        # 计算新模型的参数量
        new_params = sum(p.numel() for p in model.parameters())
        print("Total params of new model : %.2fM" % (new_params / 1e6))

        print('词表缩小为原来的:{}%'.format(round(len(new_tokenizer) / len(old_tokenizer), 4)*100))
        print('模型参数量缩小为原来的:{}%'.format(round(new_params / old_params, 4)*100))
        # for token_name, token_value in special_tokens_to_preserve.items():
        #     if token_value and token_value in new_vocab:
        #         setattr(new_tokenizer, token_name, token_value)
        # if new_tokenizer.pad_token_id is not None:
        #     model.config.pad_token_id = new_tokenizer.pad_token_id
        # if new_tokenizer.eos_token_id is not None:
        #     model.config.eos_token_id = new_tokenizer.eos_token_id  
        # if new_tokenizer.bos_token_id is not None:
        #     model.config.bos_token_id = new_tokenizer.bos_token_id
        # if hasattr(model, 'generation_config'):
        #     model.generation_config.pad_token_id = model.config.pad_token_id
        #     model.generation_config.eos_token_id = model.config.eos_token_id
        model.save_pretrained(save_path)
        new_tokenizer.save_pretrained(save_path)
        
        fin = open("new2old_token_id.json", "w", encoding="utf-8")
        fin.write(json.dumps(new2old_token_id, indent=4, ensure_ascii=False))
        fin.close()
        


class ModelVocabularyPruner(VocabularyPruner):

    def update_embeddings(self, model, new2old_token_id:dict, new_embeds, new_lm_head):
        for token_id, old_token_id in tqdm(new2old_token_id.items()):
            new_embeds.weight.data[token_id] = model.get_input_embeddings().weight.data[old_token_id]
            new_lm_head.weight.data[token_id] = model.get_output_embeddings().weight.data[old_token_id]
                
        model.set_input_embeddings(new_embeds)
        model.set_output_embeddings(new_lm_head)
        


if __name__ == "__main__":

    # 需要进行裁剪的模型路径
    model_name_or_path =  "../../models/Qwen3-8B-Base/"
    # 自己制作的词表的路
    new_tokenizer_name_or_path = 'Qwen3-8B-Base-New'
    save_path = 'Qwen3-8B-Base-Char'
    pruner = ModelVocabularyPruner()
    # 裁剪
    # pruner.prune(model_name_or_path, new_tokenizer_name_or_path, save_path)

    # 检查裁剪的模型与原模型是否一致
    pruner.check(model_name_or_path, save_path, text='长风破浪会有时')
    pruner.check_embedding(model_name_or_path, save_path, text='项伤是速')
