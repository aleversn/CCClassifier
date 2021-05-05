# %%
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
class SimDataset(Dataset):

    def __init__(self, tokenizer, file_name, padding_length=128, is_train=True, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(file_name)
        self.computed_avg_length(self.ori_list)
        if is_train:
            self.train_compose()
        else:
            self.final_list = self.ori_list
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        return ori_list
    
    def computed_avg_length(self, target):
        sum = []
        for item in target:
            sum.append(len(item))
        avg = np.average(sum)
        mid = np.median(sum)
        max = np.max(sum)
        min = np.min(sum)
        print('\navg: {}, median: {}, max: {}, min: {}\n'.format(avg, mid, max, min)) 
        return avg, mid, max, min
    
    def train_compose(self):
        self.final_list = []
        for line in self.ori_list:
            line = line.strip().split('\t')
            self.final_list.append({
                "src": line[0],
                "tgt": line[1],
                "label": 1
            })
            _tgt_list = random.sample(self.ori_list, 9)
            for _tgt in _tgt_list:
                _tgt = _tgt.split('\t')[1]
                self.final_list.append({
                    "src": line[0],
                    "tgt": _tgt,
                    "label": 1 if _tgt == line[1] else 0
                })
                self.final_list.append({
                    "src": line[0],
                    "tgt": line[1],
                    "label": 1
                })
        random.shuffle(self.final_list)
        return self.final_list
    
    def __getitem__(self, idx):
        line = self.final_list[idx]
        s1, s2, label = line['src'], line['tgt'], line['label']
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type_ids,
            "label": float(label)
        }
    
    def __len__(self):
        return len(self.final_list)

class EvalSim(Dataset):
    '''
    src_file_name: 要预测的数据集
    target_file_name: 标准问字典
    '''
    def __init__(self, tokenizer, src_file_name, target_file_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list, self.tgt_list = self.load_train(target_file_name)
        self.src_list = self.src_init(src_file_name)
        self.eval_list = []
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[len(ori_list) - 1] == '':
            ori_list = ori_list[:len(ori_list) - 1]
        tgt_list = [item.strip() for item in ori_list]
        return ori_list, tgt_list
    
    def src_init(self, src_file_name):
        with open(src_file_name, encoding='utf-8') as f:
            src_list = f.read().split('\n')
        if src_list[len(src_list) - 1] == '':
            src_list = src_list[:len(src_list) - 1]
        src_list = [item.split('\t')[0].strip() for item in src_list]
        return src_list
    
    def computed_eval_set(self, src):
        self.eval_list = []
        for line in self.ori_list:
            line = line.strip()
            self.eval_list.append({
                "src": src,
                "tgt": line,
                "label": -1
            })
        return self.eval_list
    
    def __getitem__(self, idx):
        line = self.eval_list[idx]
        s1, s2, label = line['src'], line['tgt'], line['label']
        T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type_ids,
            "label": float(label)
        }
    
    def __len__(self):
        return len(self.eval_list)

class PredSim(EvalSim):
    def __init__(self, tokenizer, target_file_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(target_file_name)
        self.eval_list = []
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[-1] == '':
            ori_list = ori_list[:-1]
        return ori_list