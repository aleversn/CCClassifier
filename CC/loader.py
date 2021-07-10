import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from CC.ICCStandard import IDataLoader
from CC._dataloaders import *

class AutoDataloader(IDataLoader):

    def __init__(self, tokenizer, data_name, model_type="bert", padding_length=50):
        self.result_set = None
        self.model_type = model_type
        model_list_1 = ['bert']
        if data_name == 'cls':
            if model_type in model_list_1:
                self.training_set = ClsDataset(tokenizer, './dataset/THUNews_proceed', padding_length=padding_length, split_ratio=[0, 0.8], is_train=True)
                self.dev_set = ClsDataset(tokenizer, './dataset/THUNews_proceed', padding_length=padding_length, split_ratio=[0.8, 0.9], shuffle=False)
                self.test_set = ClsDataset(tokenizer, './dataset/THUNews_proceed', padding_length=padding_length, split_ratio=[0.9, 1], shuffle=False)
        elif data_name == 'sccls':
            if model_type in model_list_1:
                self.training_set = SCClsDataset(tokenizer, './dataset/A7.dev.txt', padding_length=padding_length)
                self.dev_set = SCClsDataset(tokenizer, './dataset/A7.dev.txt', padding_length=padding_length, shuffle=False)
                self.test_set = SCClsDataset(tokenizer, './dataset/A7.dev.txt', padding_length=padding_length, shuffle=False)
    
    def __call__(self, batch_size=16, batch_size_eval=64):
        dataiter = DataLoader(self.training_set, batch_size=batch_size)
        dataiter_eval = DataLoader(self.dev_set, batch_size=batch_size_eval)
        dataiter_test = DataLoader(self.test_set, batch_size=batch_size_eval)
        return {
                "dataiter": dataiter,
                "dataiter_eval": dataiter_eval,
                "dataiter_test": dataiter_test
            }