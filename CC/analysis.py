import os
import datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr
from CC.ICCStandard import IAnalysis

class Analysis(IAnalysis):

    def __init__(self):
        print(0)
    
    @staticmethod
    def WriteSDC(name, info):
        with open("./log/{}.txt".format(name), mode="a+", encoding="utf-8") as f:
            f.write(info)
    
    @staticmethod
    def Evaluation(src_file_name, gold_file_name, tgt_file_name):
        with open(src_file_name, encoding='utf-8') as f:
            src_list = f.read().split('\n')
        if src_list[len(src_list) - 1] == '':
            src_list = src_list[:len(src_list) - 1]
        with open(gold_file_name, encoding='utf-8') as f:
            gold_list = f.read().split('\n')
        if gold_list[len(gold_list) - 1] == '':
            gold_list = gold_list[:len(gold_list) - 1]
        with open(tgt_file_name, encoding='utf-8') as f:
            tgt_list = f.read().split('\n')
        if tgt_list[len(tgt_list) - 1] == '':
            tgt_list = tgt_list[:len(tgt_list) - 1]
        
        tp = 0
        fp = 0
        fn = len(src_list)
        for idx, _ in enumerate(src_list):
            src, pred = src_list[idx].split('\t')
            gold = gold_list[idx].split('\t')[1]
            if pred == gold:
                tp += 1
            elif pred in gold:
                tp += 1
            elif gold in pred:
                tp += 1
            else:
                fp += 1
        
        p = tp / (tp + fp)
        r = tp / fn
        f1 = 2 * p * r / (p + r)
        
        return p, r, f1
    
    @staticmethod
    def DiffOutput(src_file_name, gold_file_name, save_file_name):
        with open(src_file_name, encoding='utf-8') as f:
            src_list = f.read().split('\n')
        if src_list[len(src_list) - 1] == '':
            src_list = src_list[:len(src_list) - 1]
        with open(gold_file_name, encoding='utf-8') as f:
            gold_list = f.read().split('\n')
        if gold_list[len(gold_list) - 1] == '':
            gold_list = gold_list[:len(gold_list) - 1]
        
        result = '原文\t预测\t人工标注\n'
        for idx, _ in enumerate(src_list):
            src, pred = src_list[idx].split('\t')
            gold = gold_list[idx].split('\t')[1]
            if pred != gold:
                result += '{}\t{}\t{}\n'.format(src, pred, gold)
        
        with open(save_file_name, mode='w+') as f:
            f.write(result)
    
    @staticmethod
    def heatmap(data):
        return ValueError('')
    
    @staticmethod
    def save_xy(X, Y, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        for i in range(len(X)):
            result += '{}\t{}\n'.format(X[i], Y[i])
        with open('{}/predict_gold.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result)
    
    @staticmethod
    def save_same_row_list(dir, file_name, **args):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        dicts = []
        for key in args.keys():
            dicts.append(key)
            result = key if result == '' else result + '\t{}'.format(key)
        length = len(args[dicts[0]])
        result += '\n'
        for i in range(length):
            t = True
            for key in args.keys():
                result += str(args[key][i]) if t else '\t{}'.format(args[key][i])
                t = False
            result += '\n'
        with open('{}/{}.csv'.format(dir, file_name), encoding='utf-8', mode='w+') as f:
            f.write(result)
