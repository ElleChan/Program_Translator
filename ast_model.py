import ast_parser as ap
import torch
from torch import nn
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
#              https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch
class EncoderModel(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_layer, dim_output):
         super(EncoderModel, self).__init__()

         self.dim_input = dim_input
         self.dim_hidden = dim_hidden
         self.dim_layer = dim_layer
         self.dim_output = dim_output

         self.embedding = nn.Embedding(dim_input, dim_hidden)     # Creates embedding matrix.
         self.gru = nn.GRU(dim_hidden, dim_hidden)                 # Kind of RNN, akin to LSTM

    def forward(self, input_vector, hidden_vector):
        embedded = self.embedding(input_vector).view(1, 1, -1)
        output_vector, hidden_vector = self.gru(embedded, hidden_vector)
        return (output_vector, hidden_vector)
        
    def initialize_hidden(self):
        return torch.zeros(1, 1, self.dim_hidden)           # 3D matrix with 1 matrix of dim_hidden items.

    
e = EncoderModel(12,10,3,4)
h0 = e.initialize_hidden()
input_v = torch.tensor([3])
(output, h0) = e.forward(input_v, h0)
input_v = torch.tensor([11])
(output, h0) = e.forward(input_v, h0)
print(output, '\n', h0)
