# %%
import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
class ClsDataset(Dataset):

    def __init__(self, tokenizer, dir_name, padding_length=128, split_ratio=[0, 0.8], is_train=False, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.split_ratio = split_ratio
        self.ori_list, self.tags_list = self.load_train(dir_name)
        if is_train:
            self.final_list = self.train_compose()
        else:
            self.final_list = self.eval_compose()
        if shuffle:
            random.shuffle(self.final_list)
    
    def load_train(self, dir_name):
        with open(os.path.join(dir_name, 'tags_list.csv'), encoding='utf-8') as f:
            tags_list = f.read().split('\n')
        if tags_list[-1] == '':
            tags_list = tags_list[:-1]
        ori_list = [i for i in range(len(tags_list))]
        files = os.listdir(dir_name)
        self.maxium = 0
        for file in files:
            if file == 'tags_list.csv':
                continue
            with open(os.path.join(dir_name, file), encoding='utf-8') as f:
                cur_list = f.read().split('\n')
            if cur_list[-1] == '':
                cur_list = cur_list[:-1]
            idx = int(cur_list[0].split('\t')[0])
            if self.maxium < len(cur_list):
                self.maxium = len(cur_list)
            cur_list = cur_list[int(self.split_ratio[0] * len(cur_list)):int(self.split_ratio[1] * len(cur_list))]
            print('{} length: {}\n'.format(tags_list[idx], len(cur_list)))
            self.computed_avg_length(cur_list)
            ori_list[idx] = cur_list
            
        return ori_list, tags_list
    
    def train_compose(self):
        final_list = []
        for unit in self.ori_list:
            final_list += unit
            addon = int(self.maxium / len(unit))
            for i in range(addon):
                final_list += unit
        return final_list
    
    def eval_compose(self):
        final_list = []
        for unit in self.ori_list:
            final_list += unit
        return final_list
    
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
    
    def __getitem__(self, idx):
        line = self.final_list[idx].strip().split('\t')
        s, label = line[1], line[0]
        T = self.tokenizer(s, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type_ids,
            "label": int(label)
        }
    
    def __len__(self):
        return len(self.final_list)

class SCClsDataset(Dataset):
    
    def __init__(self, tokenizer, file_name, padding_length=128, mode='tsv', shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.mode = mode
        self.ori_list = self.load_train(file_name)
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[-1] == '':
            ori_list = ori_list[:-1]
        return ori_list
    
    def __getitem__(self, idx):
        if self.mode == 'tsv':
            line = self.ori_list[idx].strip().split('\t')
            s, label = line[0], line[1]
        else:
            line = json.loads(self.ori_list[idx])
            s, label = line['text1'], line['label']
        
        T = self.tokenizer(s, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "token_type_ids": token_type_ids,
            "label": int(label)
        }
    
    def __len__(self):
        return len(self.ori_list)