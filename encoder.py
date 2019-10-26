import ast_parser as ap
import torch
from torch import nn
from functools import reduce
from languages import ASTNumbering
from numpy.random import choice


class AstModel:
    def __init__(self, paths):
        if isinstance(paths, str):
            self.data = ap.parseAST(paths)
        elif isinstance(paths, list):
            self.data = [ap.parseAST(x) for x in paths]
            self.data = reduce(lambda x, y: x + y, self.data)
        else:
            raise TypeError('AstModel() expects a string or a list')
        self.java_language = ASTNumbering('Java')
        self.cs_language = ASTNumbering('C#')
        for tree in self.data:
            self.java_language.add_ast(tree['java_ast'])
            self.cs_language.add_ast(tree['cs_ast'])

    def train(self, train_data, epocs, batch_size):
        for _ in range(epocs):
            for _ in range(batch_size):
                train = choice(train_data, batch_size)
                self._prepare_data(train['java_ast'])


    def predict(self, data):
        pass

    def _prepare_data(self, data):
        return self.java_language.create_vector(data)


# Encoder NN to predict encoding for source tree.
# Referenced: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#             https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch
#             https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53
class EncoderModel(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, layer_count=0, dropout_rate=0.0):
         super(EncoderModel, self).__init__()

         self.dim_input = dim_input
         self.dim_hidden = dim_hidden
         self.dim_output = dim_output
         self.layer_count = layer_count
         self.dropout_rate = dropout_rate

         self.gru = nn.GRU(dim_input, dim_hidden, dropout=dropout_rate)      # Kind of RNN, akin to LSTM

    # Moves the RNN forward to the next iter.
    def forward(self, input_vector, hidden_vector):
        input_vector = input_vector.float()
        hidden_vector = hidden_vector.float()
        output_vector, hidden_vector = self.gru(input_vector.view(1,1,-1), hidden_vector)
        return output_vector, hidden_vector
        
    def initialize_hidden(self):
        return torch.zeros(1, 1, self.dim_hidden)                               # Flattened 3D matrix of dim_hidden items.


if __name__ == '__main__':
    e = EncoderModel(1, 1, 3, 4)
    e.train()
    h0 = e.initialize_hidden()
    input_v = torch.tensor([1])
    (output, h0) = e.forward(input_v, h0)
    print(output, "\n", h0)
    e.eval()
    h0 = e.initialize_hidden()
    input_v = torch.tensor([3])
    (output, h0) = e.forward(input_v, h0)
    print(output, "\n", h0)