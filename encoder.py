import torch
from torch import nn
from math import sqrt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




# Encoder NN to predict encoding for source tree.
# Referenced: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#             https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch
#             https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53
#             https://github.com/IBM/pytorch-seq2seq/
#             https://blog.floydhub.com/gru-with-pytorch/            
#	      https://discuss.pytorch.org/t/runtimeerror-input-must-have-3-dimensions-got-2/36974/8
#             https://pytorch.org/docs/stable/nn.html
#             https://medium.com/@Petuum/embeddings-a-matrix-of-meaning-4de877c9aa27
#	      https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-for-argument-2-mat2/49849/2
class EncoderModel(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, layer_count=0, dropout_rate=0.0):
        super(EncoderModel, self).__init__()

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.layer_count = layer_count
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(dim_output, dim_input)				      # Produces matrix of shape output_dim,output_dim
        self.gru = nn.GRU(dim_input, dim_hidden, batch_first=True, dropout=dropout_rate)      # Kind of RNN, akin to LSTM
        self.fc = nn.Linear(dim_hidden, dim_output)
        self.relu = nn.ReLU()

    # Moves the RNN forward to the next iter.
    def forward(self, input_vector, hidden_vector, shape):
        #embedded = self.embedding(input_vector).view(1,self.dim_input,self.dim_input)
        embedded = input_vector.view(1, 1, self.dim_input)
        embedded = embedded.float()		      
        print("\t\t\tEmbedded matrix and shape:", embedded, embedded.size())
        #embedded = pack_padded_sequence(embedded, shape, enforce_sorted=False, batch_first=True)
        output_vector, hidden_vector = self.gru(embedded, hidden_vector)
        #output_vector, _ = pad_packed_sequence(output_vector, batch_first=True)
        output_vector = self.fc(self.relu(output_vector))
        return output_vector, hidden_vector
        
    def initialize_hidden(self):
        return torch.zeros(1, 1, self.dim_hidden)                               # 1, batch_size, hidden_size


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout):
        super(TransformerModel, self).__init__()
        layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer = nn.TransformerEncoder(layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()
        self.src_mask = None

    def init_weights(self):
        self.encoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            self.src_mask = self._generate_mask(len(x)).to(x.device)
        x = self.encoder(x) * sqrt(self.ninp)
        output = self.transformer(x, self.src_mask)
        return self.decoder(output)

    def _generate_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
