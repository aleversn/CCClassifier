import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from CC.ICCStandard import IPredict
from CC.model import AutoModel
from CC.loader import AutoDataloader
from CC.analysis import Analysis
from tqdm import tqdm
from torch.autograd import Variable

class Predictor(IPredict):

    def __init__(self, tokenizer, model_dir, padding_length=128, resume_path=False, num_labels=2, gpu=[0]):
        self.tokenizer = tokenizer
        self.model_init(tokenizer, model_dir, num_labels)
        self.padding_length = padding_length

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.load_state_dict(model_dict)
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)
    
    def model_init(self, tokenizer, model_dir, num_labels):
        a = AutoModel(tokenizer, model_dir, num_labels)
        print('AutoModel Choose Model: {}\n'.format(a.model_name))
        self.model_cuda = False
        self.config = a.config
        self.model = a()
    
    def __call__(self, X):
        return self.predict(X)
    
    def data_process(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]
        input_ids = []
        attn_mask = []
        token_type_ids = []
        label = []
        for sentence in sentences:
            T = self.tokenizer(sentence, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
            input_ids.append(torch.tensor(T['input_ids']))
            attn_mask.append(torch.tensor(T['attention_mask']))
            token_type_ids.append(torch.tensor(T['token_type_ids']))
            label.append(torch.tensor(0))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn_mask),
            "token_type_ids": torch.stack(token_type_ids),
            "label": torch.stack(label)
        }
    
    def predict(self, X):
        with torch.no_grad():
            it = self.data_process(X)

            for key in it.keys():
                it[key] = self.cuda(it[key])

            loss, pred = self.model(**it)
            loss = loss.mean()

            pred = F.softmax(pred, dim=-1)
            val, p = pred.topk(5)
            
            return val.tolist(), p.tolist()
    
    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX