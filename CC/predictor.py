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
from CC._dataloaders import PredSim
from CC.analysis import Analysis
from tqdm import tqdm
from torch.autograd import Variable

class Predictor(IPredict):

    def __init__(self, tokenizer, model_dir, target_file_name, padding_length=128, resume_path=False, batch_size=64, gpu=[0]):
        self.tokenizer = tokenizer
        self.model_init(tokenizer, model_dir)
        self.padding_length = padding_length
        self.dataloader_init(tokenizer, target_file_name, batch_size, padding_length)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.load_state_dict(model_dict)
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)
    
    def model_init(self, tokenizer, model_dir):
        a = AutoModel(tokenizer, model_dir)
        print('AutoModel Choose Model: {}\n'.format(a.model_name))
        self.model_cuda = False
        self.config = a.config
        self.model = a()
    
    def dataloader_init(self, tokenizer, target_file_name, batch_size, padding_length):
        result_dataset = PredSim(tokenizer, target_file_name, padding_length)
        self.result_loader = DataLoader(result_dataset, batch_size=batch_size)
    
    def __call__(self, X):
        return self.predict(X)
    
    def predict(self, X):
        with torch.no_grad():
            result = []
            self.model.eval()
            result_iter = self.result_loader
            result_iter.dataset.computed_eval_set(X)
            result_iter = self.result_loader
            src_result = torch.tensor([]).cuda()

            for it in result_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                loss, pred = self.model(**it)
                loss = loss.mean()

                pred = pred[:, 1]
                src_result = torch.cat([src_result, pred], -1)
            
            pred_result = src_result.sort(dim=-1, descending=True)[1].tolist()
            result = [self.result_loader.dataset.ori_list[idx] for idx in pred_result]
            
            return result
    
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