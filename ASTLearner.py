from languages import ASTNumbering
import numpy as np
from numpy.random import choice
from decoder import DecoderModel
from encoder import EncoderModel, TransformerModel
import torch

class ASTModel:
    def __init__(self, java_language, cs_language, loss_func=torch.nn.NLLLoss()):
        self.java_language = java_language
        self.cs_language = cs_language
        self.encoder = EncoderModel(java_language.max_length, java_language.max_length, 1)
        self.decoder = DecoderModel(java_language.max_length, java_language.max_length, 1)
        self.loss = loss_func
        self._hidden_encoder = self.encoder.initialize_hidden()
        self._hidden_decoder = 0
        
    def train(self, train_data, epochs, batch_size, verbose=False):
        #train_data = [self._prepare_data(x) for x in train_data]
        java_train = []
        cs_train = []
        
        for i in train_data:
            java_train.append(i['java_ast'])
            cs_train.append(i['cs_ast'])

        self.fit(epochs, java_train, cs_train, batch_size, 0.1, verbose)

    def fit(self, epochs, java_train, cs_train, batch_size, learning_rate, verbose):
        for _ in range(epochs):
            matrix, _, lengths = self._create_batch(java_train, batch_size)
            print(matrix, lengths, sep='\n\n')
            outputs = []
            o, self._hidden_encoder = self.encoder.forward(matrix, self._hidden_encoder, lengths)
            print(o)

    def predict(self, data):
        pass
        
    def _create_batch(self, x, batch_size):
        vectors = choice(len(x), size=batch_size)
        return vectors

    def _prepare_data(self, data):
        return (self.java_language.create_vector(data['java_ast']),
                 self.cs_language.create_vector(data['cs_ast']))

    def eval(self, x, y):
        pass


class ASTModelTransformer(ASTModel):
    def __init__(self, java_language, cs_language, loss_func=torch.nn.NLLLoss(), heads=1, dim_feedforward=2048,
                 dropout=0.1, activation='relu', number_encoder_layers=1, number_decoder_layers=1):
        print("Creating Transformer Model")
        super().__init__(java_language, cs_language, loss_func)
        self.transformer = torch.nn.Transformer(java_language.count, heads, number_encoder_layers,
                                                number_decoder_layers, dim_feedforward, dropout, activation)
        self.transformer = TransformerModel(java_language.count, java_language.max_length, heads, dim_feedforward, number_encoder_layers, dropout)



    def fit(self, epochs, java_train, cs_train, batch_size, learning_rate, verbose):
        """
        Meant to be called using train, given 2 arrays of vectors chooses random elements
        :param learning_rate:
        :param epochs:
        :param java_train:
        :param cs_train:
        :param batch_size:
        :param verbose:
        :return:
        """
        print("Starting training")
        self.transformer.train()
        loss = 0
        size = self.java_language.count
        
        temp = [torch.zeros(size) for _ in java_train]
        for vector, original in zip(temp, java_train):
            vector[:original.shape[-1]] = original[0, :]
        java_train = temp
        
        temp = [torch.zeros(size) for _ in java_train]
        for vector, original in zip(temp, cs_train):
            vector[:original.shape[-1]] = original[0, :]
        cs_train = temp
        print(cs_train[0].shape)
        optimizer = torch.optim.SGD(self.transformer.parameters(), lr=learning_rate)
        total_loss = 0
        losses = []
        for run_number in range(epochs):
            for i in self._create_batch(java_train, batch_size):
                java_vector = java_train[i]
                java_vector = java_vector.view((1, 1, -1))

                input_msk = None
                # input_seq = java_vector.transpose(0,1)
                # input_msk = (input_seq != -1).unsqueeze(1)

                cs_vector = cs_train[i]
                cs_vector = cs_vector.view((1, 1, -1))

                target_msk = None
                # target_seq = cs_vector.transpose(0,1)
                # target_msk = (input_seq != -1).unsqueeze(1)

                # temp_size = target_seq.size(-1)

                # temp_mask = np.triu(np.ones((1, temp_size, temp_size)))
                # temp_mask = (torch.from_numpy(temp_mask) == 0).unsqueeze(1)

                # target_msk = target_msk & temp_mask

                target_input = cs_vector[:, :, :-1]
                target = cs_vector[:, :, 1:].contiguous().view(-1)
                optimizer.zero_grad()
                print(java_vector.shape, cs_vector.shape)
                output = self.transformer(java_vector)#, cs_vector, input_msk, target_msk)
                print(output)
                print(output.shape)

                print(target.shape)
                loss = self.loss(output.view(-1, output.size(-1)), target.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
            print('\nepoch {} | current loss {}'.format(run_number, total_loss / batch_size))
            losses.append(total_loss / batch_size)
            total_loss = 0
        return losses

    def predict(self, x):
        return self.transformer(x)

    def eval(self, x, y):
        self.transformer.eval()
        total_loss = 0
        for vector in x:
            output = self.transformer(vector)
            output = output.view(-1, self.java_language.max_length)
            total_loss += len(vector) * self.loss(output, y)
        return total_loss / (len(x) - 1)
