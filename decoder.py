from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Reference: https://github.com/IBM/pytorch-seq2seq/
class DecoderModel(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderModel, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, shape):
        embedded = pack_padded_sequence(input, shape, batch_first=True, enforce_sorted=False)
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        embedded = pad_packed_sequence(embedded, batch_first=True)
        output = self.softmax(self.out(embedded[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)