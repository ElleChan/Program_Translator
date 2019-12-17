from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Reference: https://github.com/IBM/pytorch-seq2seq/
#            https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class DecoderModel(nn.Module):
    def __init__(self,dim_input, dim_hidden, dim_output, dropout_rate=0.0):
        super(DecoderModel, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output

        self.embedding = nn.Embedding(dim_output, dim_input)
        self.gru = nn.GRU(dim_input, dim_hidden, batch_first=True, dropout=dropout_rate)
        self.out = nn.Linear(dim_hidden, dim_output)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, input_vector, hidden, shape):
        #embedded = self.embedding(input_vector).view(1,1,-1)
        #embedded = pack_padded_sequence(embedded, shape, batch_first=True, enforce_sorted=False)
        #embedded = F.relu(input_vector)
        output, hidden = self.gru(input_vector, hidden)
        #output = pad_packed_sequence(output, batch_first=True)
        output = self.softmax(self.out(output))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.dim_hidden)
