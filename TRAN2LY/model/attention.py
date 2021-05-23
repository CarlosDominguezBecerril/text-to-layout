import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, hidden_size]
        #encoder_outputs = [batch size, src len, hidden_size]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #hidden = [batch size, src_len, hidden_size]

        energy = self.attn(torch.cat((hidden, encoder_outputs), dim = 2))
        #energy = [batch size, src_len, hidden_size]

        attention = self.v(energy).squeeze(2)
        #attention= [batch size, src_len]
        
        return F.softmax(attention, dim=1)