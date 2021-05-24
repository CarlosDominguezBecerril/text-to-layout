import sys

import torch
import torch.nn as nn

from transformers import *
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

class EncoderBERT(nn.Module):
    def __init__(self, pretrained_path=None, freeze=False):
        super(EncoderBERT, self).__init__()

        self.pretrained_path = pretrained_path # The path of the pretrained model (None if BERT-base)
        self.freeze = freeze 
        
        # Build the model
        self.model = None
        if self.pretrained_path == None:
            self.model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.pretrained_path)


    def forward(self, input_ids, attention_mask):
        """
        Applies a BERT transformer to an input sequence
        Args:
            input_ids (batch, seq_len): tensor containing the features of the input sequence.
            attention_mask (batch, seq_len)
        Returns:

        """

        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Return the [CLS] token embedding
        cls = outputs.hidden_states[-1].permute(1, 0, 2)[0]
        # cls [batch_size, 768]

        return cls


