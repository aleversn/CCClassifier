import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

class Bert(nn.Module):

    def __init__(self, tokenizer, pretrained_dir, num_labels=2):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=num_labels)
        self.config = self.model.config
        self.tokenizer = tokenizer
    
    def forward(self, **args):
        fct_loss = nn.CrossEntropyLoss()
        outputs = self.model(input_ids=args['input_ids'], attention_mask=args['attention_mask'], token_type_ids=args['token_type_ids'])

        logits = outputs[0]

        loss = fct_loss(logits.view(-1, self.config.num_labels), args['label'].view(-1))

        return loss, logits