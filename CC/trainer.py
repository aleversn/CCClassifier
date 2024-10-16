import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from CC.ICCStandard import ITrainer
from CC.model import AutoModel
from CC.loader import AutoDataloader
from CC.analysis import Analysis
from tqdm import tqdm
from torch.autograd import Variable
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer(ITrainer):

    def __init__(self, tokenizer, model_dir, dataset_name, padding_length=50, num_labels=2, batch_size=16, batch_size_eval=64):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.num_labels = num_labels
        self.model_init(tokenizer, model_dir, num_labels)
        self.padding_length = padding_length
        self.dataloader_init(tokenizer, dataset_name, self.config['model_type'], padding_length, batch_size, batch_size_eval)
    
    def model_init(self, tokenizer, model_dir, num_labels):
        a = AutoModel(tokenizer, model_dir, num_labels)
        print('AutoModel Choose Model: {}\n'.format(a.model_name))
        self.model_cuda = False
        self.config = a.config
        self.model = a()

    def dataloader_init(self, tokenizer, data_name, model_type, padding_length, batch_size=16, batch_size_eval=64):
        d = AutoDataloader(tokenizer, data_name, model_type, padding_length)
        it = d(batch_size, batch_size_eval)
        self.train_loader = it['dataiter']
        self.eval_loader = it['dataiter_eval']
        self.test_loader = it['dataiter_test']
    
    def save_pretrained(self, resume_path, save_name, gpu=[0, 1, 2, 3]):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)

        print('Accessing Resume PATH: {} ...\n'.format(resume_path))
        model_dict = torch.load(resume_path).module.state_dict()
        self.model.module.load_state_dict(model_dict)
        self.model.to(device)
        self.model.module.model.save_pretrained(save_name)
    
    def __call__(self, resume_path=False, num_epochs=30, lr=5e-5, gpu=[0, 1, 2, 3], score_fitting=False):
        self.train(resume_path, num_epochs, lr, gpu, score_fitting)

    def train(self, resume_path=False, num_epochs=10, lr=5e-5, gpu=[0, 1, 2, 3], is_eval='train/eval', eval_mode='dev', fp16=False, fp16_opt_level='O1'):
        '''
        is_eval: decide whether to eval, True - both training and evaluating; False - only training.
        eval_mode: 'dev' or 'test'.
        fp16_opt_level: For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, 80000)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=fp16_opt_level)
        
        if torch.cuda.is_available() and self.model_cuda == False:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
            self.model_cuda = True
            self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
            self.model.to(device)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, 80000)
        
        Epoch_loss = []
        Epoch_acc = []
        Epoch_loss_eval = []
        Epoch_acc_eval = []
        for epoch in range(num_epochs):
            train_count = 0
            train_result = []
            train_loss = []
            # self.train_loader.dataset.train_compose()
            train_iter = tqdm(self.train_loader)
            self.model.train()
            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])
                self.model.zero_grad()
                
                loss, pred = self.model(**it)
                loss = loss.mean()

                optimizer.zero_grad()
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                train_loss.append(loss.data.item())

                train_count += 1
                p = pred.max(-1)[1]
                train_result += (p == it['label']).int().tolist()

                train_iter.set_description('Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(train_loss=np.mean(train_loss), train_acc=np.mean(train_result))
            
            Epoch_loss.append(np.mean(train_loss))
            Epoch_acc.append(np.mean(train_result))
            
            _dir = './log/{}/{}'.format(self.dataset_name, self.config["model_type"])
            Analysis.save_same_row_list(_dir, 'train_log', loss=Epoch_loss, acc=Epoch_acc)
            if resume_path == False:
                self.save_model(epoch, 0)
            else:
                self.save_model(epoch, int(resume_path.split('/')[-1].split('_')[1].split('.')[0]))
            
            if is_eval == True:
                acc, eval_loss, _ = self.eval(epoch, num_epochs, eval_mode=eval_mode, gpu=gpu)
                Epoch_loss_eval.append(eval_loss)
                Epoch_acc_eval.append(acc)
                Analysis.save_same_row_list(_dir, 'eval_log', loss=Epoch_loss_eval, acc=Epoch_acc_eval)

    def save_model(self, epoch, save_offset=0):
        _dir = './model/{}/{}'.format(self.dataset_name, self.config["model_type"])
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        torch.save(self.model, '{}/epoch_{}.pth'.format(_dir, epoch + 1 + save_offset))

    def eval(self, epoch, num_epochs, resume_path=False, eval_mode='dev', gpu=[0, 1, 2, 3]):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available() and self.model_cuda == False:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
            self.model_cuda = True
            self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
            self.model.to(device)
        
        with torch.no_grad():
            eval_count = 0
            eval_result = []
            eval_pred = []
            eval_gold = []
            eval_loss = []
            self.model.eval()
            eval_iter = tqdm(self.eval_loader) if eval_mode == 'dev' else tqdm(self.test_loader)
            tp = {}
            fp = {}
            fn = {}
            for i in range(self.num_labels):
                tp[i] = 0
                fp[i] = 0
                fn[i] = 0
            for it in eval_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                loss, pred = self.model(**it)
                loss = loss.mean()

                eval_loss.append(loss.data.item())

                eval_count += 1

                p = pred.max(-1)[1]
                eval_pred += p.tolist()
                eval_gold += it['label'].tolist()
                for pred, gold in zip(eval_pred, eval_gold):
                    if pred == gold:
                        tp[gold] += 1
                    else:
                        fn[gold] += 1
                        fp[pred] += 1
                
                result_dict = {}
                for i in range(self.num_labels):
                    result_dict[str(i)] = {
                        'precision': tp[i] / (tp[i] + fp[i]),
                        'recall': tp[i] / (tp[i] + fn[i]),
                        'f1': 2 * tp[i] / (2*tp[i] + fp[i] + fn[i]) if tp[i] != 0 else 0,
                    }
                result_dict['avg_f1'] = np.mean([result_dict[str(i)]['f1'] for k in range(self.num_labels)])
                result_dict['avg_precision'] = np.mean([result_dict[str(i)]['precision'] for k in range(self.num_labels)])
                result_dict['avg_recall'] = np.mean([result_dict[str(i)]['recall'] for k in range(self.num_labels)])

                eval_iter.set_description('Eval: {}/{}'.format(epoch + 1, num_epochs))
                eval_iter.set_postfix(eval_loss=np.mean(eval_loss), eval_acc=np.mean(eval_result), **result_dict)
            
            _dir = './log/{}/{}'.format(self.dataset_name, self.config["model_type"])
            Analysis.save_same_row_list(_dir, 'pred_gold', eval_pred=eval_pred, eval_gold=eval_gold)
            return np.mean(eval_result), np.mean(eval_loss), result_dict
    
    def save_pred(self, save_dir, resume_path=None, gpu=[0, 1, 2, 3]):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available() and self.model_cuda == False:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
            self.model_cuda = True
            self.model.to(device)
            
        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
            self.model.to(device)
        
        with torch.no_grad():
            result = ''
            self.model.eval()
            src_iter = tqdm(self.result_loader.dataset.src_list)
            result_iter = self.result_loader
            for src in src_iter:
                result_iter.dataset.computed_eval_set(src)
                src_result = torch.tensor([]).cuda()

                for it in result_iter:
                    for key in it.keys():
                        it[key] = self.cuda(it[key])

                    loss, pred = self.model(**it)
                    loss = loss.mean()

                    pred = pred[:, 1]
                    src_result = torch.cat([src_result, pred], -1)
                
                pred_result = src_result.max(-1)[1].tolist()
                result += '{}\t{}\n'.format(src, self.result_loader.dataset.tgt_list[pred_result])

                src_iter.set_description('Preding')
                src_iter.set_postfix(src=src, tgt=self.result_loader.dataset.tgt_list[pred_result])
            
            with open(os.path.join(save_dir, 'pred_result.csv'), mode='w+') as f:
                f.write(result)
            
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