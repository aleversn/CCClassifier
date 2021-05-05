import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

class Bert(nn.Module):

    def __init__(self, tokenizer, pretrained_dir):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        self.tokenizer = tokenizer
    
    def forward(self, **args):
        fct_loss = nn.BCELoss()
        outputs = self.model(input_ids=args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'])

        logits = outputs[0]
        pred = F.softmax(logits, dim=-1)

        loss = fct_loss(pred[:, 1], args['label'].float().view(-1))

        return loss, pred