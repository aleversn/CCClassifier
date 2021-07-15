# %%
from CC.process import *

process_data('./dataset/THUCNews', './dataset/THUNews_proceed')


# %%
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
from CC.trainer import *
from transformers import BertTokenizer

# %%
tokenizer = BertTokenizer.from_pretrained('model/chinese_wwm_ext')
trainer = Trainer(tokenizer, model_dir='model/chinese_wwm_ext', dataset_name='cls', padding_length=400, num_labels=14, batch_size=32, batch_size_eval=500)

# %%
# Common Training
trainer.train(num_epochs=30, lr=1e-5, resume_path='./model/cls/bert/epoch_1.pth', gpu=[0, 1], eval_mode='test', is_eval=False)

# %%
trainer.eval(0, 0, resume_path='./model/cls/bert/epoch_1.pth', gpu=[0, 1], eval_mode='dev')

# %%
from CC.predictor import *
from transformers import BertTokenizer

# %%
tokenizer = BertTokenizer.from_pretrained('model/chinese_wwm_ext')
pred = Predictor(tokenizer, model_dir='model/chinese_wwm_ext', padding_length=400, resume_path='./model/cls/bert/epoch_1.pth', num_labels=14, gpu=[0, 1])

# %%
pred('格库电铁（青海境内）110千伏外部供电工程首条线路成功带电 西宁9月1日电 (孙睿 马红娟)记者1日从青海省海西州格尔木市官方获悉，格尔木至库尔勒电气化铁路(青海境内)110千伏外部供电工程首条线路110千伏土砂牵线于8月31日成功带电。格库电铁(青海境内)110千伏外部供电工程线路位于青海省西部及新疆东南部。东起青海省格尔木市，沿昆仑山北麓，柴达木盆地南缘西行，与省道318线、315国道伴行，经乌图美仁、甘森、花土沟至茫崖石棉矿，进入新疆境内线路穿越阿尔金山，西抵库尔勒市。海西供电公司建管范围起于格尔木市，途经乌图美仁、甘森、花土沟至茫崖石棉矿，进入新疆境内。格库电铁(青海境内)110千伏外部供电工程是格库铁路配套工程项目，工程的建成将满足格库铁路8个牵引变电站供电需求，为格库铁路的运行持续提供可靠、稳定电力保障。工程建设线路全长724公里，包括16条线路，杆塔基数2337基，分为8个标段施工，目前已完成全线基础、铁塔施工，导线架设已基本贯通，剩余施工任务正在加紧施工当中，计划9月中旬全面投运完成。图为海西供电公司专业人员正在花土沟至油沙山段施工现场进行相关调试准备工作。马红娟 摄海西供电公司格库电铁110千伏外部供电工程(青海境内)业主项目部经理高明说：“110千伏外部供电工程首条线路110千伏土砂牵线成功带电，标志着格库电铁110千伏外部供电工程青海段已全面进入竣工投运阶段，为9月30日格库铁路顺利通车奠定良好基础，后续还将有15条线路陆续投运，9月中旬前电网项目全面完工。”为保证工程顺利建成投运，自2019年3月开工以来，工程克服了现场施工点多、线长、范围广以及施工人数众多的管理困难，攻克高山、荒漠、沼泽、盐碱地等不良地质因素，克服高原缺氧、沙尘暴、气温、蚊虫叮咬等不利环境条件，通过细化施工任务目标分解、定额分析施工力量投入、“日管控、周汇报、月总结”等方式，科学系统的布置、落实了推进工程建设各项工作。据了解，格库铁路是客货共线的区域网干线，是国家“十横十纵”综合交通运输通道的组成部分，是中国又一条入疆大动脉，也是“丝绸之路经济带”与核心区交通枢纽中心规划建设的“东联西出”三大铁路通道中南通道的重要组成部分。格库电气化铁路不仅对完善青海、新疆铁路网布局具有重要意义，还将进一步完善我国内陆与中亚、地中海等地区的陆路运输通道，推进“一带一路”倡议的实现。(完)')

# %%
with open('./dataset/A7.dev.txt') as f:
    ori_list = f.read().split('\n')
if ori_list[-1] == '':
    ori_list = ori_list[:-1]
ori_list = [item.strip().split('\t') for item in ori_list]
split_num = int(len(ori_list) / 1000) + 1
pred_list = []
for i in tqdm(range(split_num)):
    cur_ori_list = ori_list[i * 1000 : (i + 1) * 1000]
    pred_list += pred([item[0] for item in cur_ori_list])

# %%
with open('./pred_gold', mode='w+') as f:
    f.write('')
with open('./pred_gold', mode='a+') as f:
    for idx, _ in enumerate(pred_list):
        f.write('{}\t{}\t{}\n'.format(pred_list[idx], ori_list[idx][1], ori_list[idx][0]))

# %%
with open('./dataset/THUNews_proceed/tags_list.csv') as f:
    tag_list = f.read().split('\n')
if tag_list[-1] == '':
    tag_list = tag_list[:-1]

i2tag = {}
for item in tag_list:
    item = item.split('\t')
    i2tag[int(item[0])] = item[1]

with open('./dataset/speedtest.csv') as f:
    ori_list = f.read().split('\n')
if ori_list[-1] == '':
    ori_list = ori_list[:-1]
ori_list = [item.strip().split(',') for item in ori_list]
iter_ = tqdm(ori_list)
for item in iter_:
    iter_.set_postfix(info=i2tag[pred(item[3])[1][0][0]])

# %%
with open('./submit.csv', mode='w+') as f:
    f.write('')
with open('./submit.csv', mode='a+') as f:
    for idx, item in enumerate(ori_list):
        f.write('{},{},{},{}\n'.format(item[0], item[1], item[2], item[3]))


# %%
from sklearn.metrics import f1_score

with open('./log/cls/bert/pred_gold.csv') as f:
    pred = []
    gold = []
    ori_list = f.read().split('\n')[1:-1]
    for item in ori_list:
        item = item.split('\t')
        pred.append(int(item[0]))
        gold.append(int(item[1]))

print(f1_score(gold, pred, average='macro'))
