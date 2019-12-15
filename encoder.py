import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




# Encoder NN to predict encoding for source tree.
# Referenced: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#             https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch
#             https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53
#             https://github.com/IBM/pytorch-seq2seq/
#             https://blog.floydhub.com/gru-with-pytorch/            
class EncoderModel(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, layer_count=0, dropout_rate=0.0):
        super(EncoderModel, self).__init__()

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.layer_count = layer_count
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(dim_output, dim_input)
        self.gru = nn.GRU(dim_input, dim_hidden, batch_first=True, dropout=dropout_rate)      # Kind of RNN, akin to LSTM
        self.fc = nn.Linear(dim_hidden, dim_output)
        self.relu = nn.ReLU()

    # Moves the RNN forward to the next iter.
    def forward(self, input_vector, hidden_vector, shape):
        embedded = self.embedding(input_vector)
        #embedded = pack_padded_sequence(embedded, shape, enforce_sorted=False, batch_first=True)
        output_vector, hidden_vector = self.gru(embedded, hidden_vector)
        #output_vector, _ = pad_packed_sequence(output_vector, batch_first=True)
        #output_vector = self.fc(self.relu(output_vector[:,-1]))
        return output_vector, hidden_vector
        
    def initialize_hidden(self):
        return torch.zeros(1, 1, self.dim_hidden)                               # 1, batch_size, hidden_size


