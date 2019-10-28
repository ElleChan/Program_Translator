from torch import nn
import torch
import torch.nn.functional as F


# Reference: https://github.com/IBM/pytorch-seq2seq/
class DecoderModel(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderModel, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        input = input.float()
        hidden = hidden.float()
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)