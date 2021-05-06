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
trainer = Trainer(tokenizer, model_dir='model/chinese_wwm_ext', dataset_name='cls', padding_length=400, num_labels=14, batch_size=64, batch_size_eval=1000)

# %%
# Common Training
trainer.train(num_epochs=30, lr=1e-5, gpu=[0, 1, 2, 3], eval_mode='test', is_eval=False)

# %%
trainer.eval(0, 0, resume_path='./model/cls/bert/epoch_1.pth', gpu=[0, 1, 2, 3], eval_mode='dev')

# %%
