from languages import ASTNumbering
import numpy as np
from numpy.random import choice
from decoder import DecoderModel
from encoder import EncoderModel
import torch

class ASTModel:
    def __init__(self, java_language, cs_language, loss_func=torch.nn.NLLLoss()):
        self.java_language = java_language
        self.cs_language = cs_language
        self.encoder = EncoderModel(java_language.max_length, java_language.max_length, 1)
        self.decoder = DecoderModel(java_language.max_length, java_language.max_length)
        self.loss = loss_func
        self._hidden_encoder = self.encoder.initialize_hidden()
        self._hidden_decoder = 0
        
    def train(self, train_data, epochs, batch_size):
        train_data = [self._prepare_data(x) for x in train_data]
        java_train = []
        cs_train = []
        
        for i in train_data:
            java_train.append(i[0])
            cs_train.append(i[1])
        for _ in range(epochs):
            matrix, lengths = self._create_batch(java_train, batch_size)
            print(matrix, lengths, sep='\n\n')
            outputs = []
            o, self._hidden_encoder = self.encoder.forward(matrix, self._hidden_encoder, lengths)
            print(o)


    def predict(self, data):
        pass
        
    def _create_batch(self, data, batch_size):
        vectors = choice(len(data), size=batch_size)
        matrix = torch.zeros(batch_size, self.java_language.max_length)
        lengths = torch.zeros(batch_size)
        counter = 0
        for i in vectors:
            print(data[i][0].shape)
            
            matrix[counter, :] = data[i][0][0, :]
            lengths[counter] = data[i][1]
            counter += 1
        return matrix, lengths

    def _prepare_data(self, data):
        return (self.java_language.create_vector(data['java_ast']),
                 self.cs_language.create_vector(data['cs_ast']))
