import ast_parser as ap
import torch
from torch import nn
import torchvision
from functools import reduce


class AstModel:
    def __init__(self, paths):
        if isinstance(paths, str):
            self.data = ap.parseAST(paths)
        else:
            self.data = [ap.parseAST(x) for x in paths]
            self.data = reduce(lambda x, y: x + y, self.data)

    def train(self):
        pass

    def predict(self, data):
        pass

    def _prepare_data(self):
        pass


# Encoder NN to predict encoding for source tree.
# Referenced: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class EncoderModel(nn.Module, n_input, n_hidden):
    def __init__():
         super(EncoderModel, self).__init__()

         self.hidden_size = hidden_size
         self.embedding = nn.Embedding(n_input, n_hidden)
         self.gru = nn.GRU(n_hidden, n_hidden)                # Kind of RNN, akin to LSTM

    def forward
        
