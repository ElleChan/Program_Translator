from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Reference: https://github.com/IBM/pytorch-seq2seq/
#            https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
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
        embedded = self.embedding(input)
        embedded = pack_padded_sequence(embedded, shape, batch_first=True, enforce_sorted=False)
        embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = pad_packed_sequence(output, batch_first=True)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)