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
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Build the model
        self.model = None
        if self.pretrained_path == None:
            self.model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.pretrained_path)


    # def forward(self, input_ids, attention_mask):
    def forward(self, captions):
        """
        Applies a BERT transformer to an input sequence
        Args:
            input_ids (batch, seq_len): tensor containing the features of the input sequence.
            attention_mask (batch, seq_len)
        Returns:

        """
        
        # Tokenize the input and prepare it for the model            
        encoding = self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        print("inputs_ids", input_ids)
        # Mover a otro lado
        if self.freeze == True:
            self.model.eval()
        else:
            self.model.train()
        

        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Return the [CLS] token embedding
        print([o.shape for o in outputs.hidden_states])
        cls = outputs.hidden_states[-1].permute(1, 0, 2)[0]
        # cls [batch_size, 768]
        # print("cls shape", cls.shape)
        return cls


# NOTE: only to test the class
# To load the BERT trained by me (caption-triples matching)

encoder = EncoderBERT(pretrained_path=None, freeze=True)

# To load the original BERT-base
#encoder = EncoderBERT(pretrained_path=None, freeze=True)
captions = ["Here we are!", "A dog is running.", "What a nice day to go to the beach!"]

cls = encoder(captions)
print(cls.size())

