# Containes modules for the Encoder/Decoder.

# Highly referenced: https://www.guru99.com/seq2seq-model.htmlhttps://www.guru99.com/seq2seq-model.html


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
#import os
#import re
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

# The individual encoder module
class Encoder(nn.Module):
      def __init__(self, dim_input, dim_hidden, dim_embed):
            super(Encoder, self).__init__()
            self.dim_input = dim_input
            self.dim_hidden = dim_hidden
            self.dim_embed = dim_embed

            self.embedding = nn.Embedding(dim_input, dim_embed)
            self.gru = nn.GRU(dim_embed, dim_hidden)

      def forward(self, input_vector):
            input_vector = input_vector.long().cuda()
            embedded = self.embedding(input_vector).view(1,self.dim_embed, self.dim_embed)
            outputs, hidden = self.gru(embedded)
            return outputs, hidden


# The individual decoder module.
class Decoder(nn.Module):
      def __init__(self, dim_output, dim_hidden, dim_embed):
            super(Decoder, self).__init__()
            self.dim_output = dim_output
            self.dim_hidden = dim_hidden
            self.dim_embed = dim_embed

            self.embedding = nn.Embedding(dim_output, dim_embed)
            self.gru = nn.GRU(dim_embed, dim_hidden)
            self.out = nn.Linear(dim_hidden, dim_output)
            self.softmax = nn.LogSoftmax(dim=1)


      def forward(self, input_vector, hidden_vector):
            input_vector = input_vector.view(1,-1)
            embedded = F.relu(self.embedding(input_vector))
            output, hidden = self.gru(embedded)
            prediction = self.softmax(self.out(output[0]))
            return prediction, hidden

# The entire encoder-decoder module.
class Seq2Seq(nn.Module):
      def __init__(self, encoder, decoder, device):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device

      def forward(self, src, target, teacher_forcing_ratio=0.5):
            input_length = src.size(0)			# Number of tokens in tree.
            batch_size = target.shape[1]
            target_length = target.shape[0]
            vocab_size = self.decoder.dim_output

            outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

            for i in range(input_length):
                encoder_output, encoder_hidden = self.encoder(src[i])    
            decoder_hidden = encoder_hidden.to(device)
            decoder_input = torch.tensor([SOS_token], device=device)
            for t in range(target_length):   
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                teacher_force = random.random() < teacher_forcing_ratio
                topv, topi = decoder_output.topk(1)
                input = (target[t] if teacher_force else topi)
                if(teacher_force == False and input.item() == EOS_token):
                    break

            return outputs


