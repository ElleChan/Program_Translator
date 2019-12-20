"""
Jael Butler and Justin Hale
CS 4860
"""
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
import ast_parser as ap
from os.path import join, realpath
import numpy as np
from languages import ASTNumbering as ast
from nn_models import Encoder, Decoder, Seq2Seq

SOS_token = 0
EOS_token = 1

# Hyperparameters
TEACHER_FORCING_RATIO = 0.5
BATCH_SIZE = 10
EPOCHS = 100
HIDDEN_SIZE=5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = ['antlr-data.json', 'itext-data.json', 'jgit-data.json', 'jts-data.json', 'lucene-data.json', 'poi-data.json']
paths = [join(realpath('.'), 'java2c#', name) for name in paths]

# Get all trees.
print("Getting all the trees")
results = [ap.parseAST(path) for path in paths]
all_results = reduce(lambda x, y: x+y, results)
print("\tTotal of", len(all_results), "trees.")

# Generate data sets
print("Creating tree classes...")
java_language = ast('Java')
cs_language = ast('CS')
for tree in all_results:
    cs_language.add_ast(tree['cs_ast'])
    java_language.add_ast(tree['java_ast'])
print("\tMax Java length and total count:", java_language.max_length, java_language.count)
print("\tMax C# length and total count:", cs_language.max_length, cs_language.count)

# Create numerical input vectors.
print("Converting trees to numerical vectors...")
dataset = []
for tree in all_results:
    cs_tree, cs_length = cs_language.create_vector(tree['cs_ast'])
    java_tree, java_length = java_language.create_vector(tree['java_ast'])
    dataset.append({'cs_ast': cs_tree, 'java_ast': java_tree})
print("\tLength of data set is:", len(dataset))
print("\tShape of first Java vector is:", dataset[0]['java_ast'], '->', dataset[0]['java_ast'].size())
print("\tShape of first C# vector is:", dataset[0]['cs_ast'], '->', dataset[0]['cs_ast'].size())

print("Splitting and shuffling dataset")
np.random.shuffle(dataset)
train, test = np.split(dataset, np.array([-1000]))
print("\tLength of training set is:", len(train))
print("\tLength of testing set is:", len(test))




# Performs one epoch calculation for a single data point for a model.
# Returns the epoch's loss.
def calcModel(model, input_vector, target_vector, model_optimizer, criterion):
     model_optimizer.zero_grad()
     input_length = input_vector.size(0)
     loss=0
     epoch_loss=0
     
     output = model(input_vector, target_vector)			# Get the model's prediction.
     num_iter = output.size(0)						# Get the length of the prediction.
     #print("Number of iterations:", num_iter)

     for ot in range(num_iter):
        loss += criterion(output[ot], target_vector[ot].long().cuda())
     loss.backward()
     model_optimizer.step()
     epoch_loss = loss.item() / num_iter				# Average the total loss for the epoch.
     return epoch_loss


# Trains the model and returns the fitted model.
# Also saves the fitted model in a save file.
def trainModel(model, train_set, epochs=10):
     model.train()

     optimizer = optim.SGD(model.parameters(), lr=0.01)			# Use SGD with learning rate of 0.01
     criterion = nn.NLLLoss()						# Use Negative Log Likelihood Loss.
     total_loss_iterations = 0		

     batch = np.random.choice(train, size=epochs)
     #batch = [tensorsFromPair(train_x, train_y, random.choice(batch_size))for i in range(epochs)]

     handle = open('loss_data.txt', 'w')

     for epoch in range(1, epochs+1):
        training_pair = batch[epoch - 1]
        print("\t\tEpoch", epoch)
        x = training_pair['java_ast']
        y = training_pair['cs_ast']

        loss = calcModel(model, x, y, optimizer, criterion)
        handle.write('%.5f\n' % loss)
        total_loss_iterations += loss
        print("\t\t\tLoss:", loss, "\tTotal loss:", total_loss_iterations)
        if epoch % 5000 == 0:
           avarage_loss= total_loss_iterations / 5000
           total_loss_iterations = 0
           print('%d %.4f' % (epochs, avarage_loss))
     #torch.save(model.state_dict(), 'mytraining.pt')
     handle.close()
     return model

def evaluate(model, test_point):
     with torch.no_grad():
        input_vector = test_point['java_ast']
        actual_vector = test_point['cs_ast']
        decoded_words = []

        output = model(input_vector, actual_vector).long()
        print('\toutput {}'.format(output), "Size:", output.size())
        for ot in range(output.size(0)):
        #for ot in range(output.size(1)):
           topv, topi = output[0][ot].topk(1)
           if topi[0].item() == EOS_token:
               decoded_words.append('<EOS>')
               break
           else:
              decoded_words.append(cs_language.convert_back[topi[0].item()])
        #output = output[0]
        #for row in output:
        #   topv, topi = output[row].topk(1)
        #   if topi[0].item() == EOS_token:
        #       decoded_words.append('End')
        #       break
        #   else:
        #      decoded_words.append(cs_language.convert_back[topi[0].item()])
     return decoded_words


def evaluateRandomly(model, test_set, number=10):
     test_points = np.random.choice(test, size=number)
     for tp in test_points:
         print('\tsource {}'.format(tp['java_ast']))
         print('\ttarget {}'.format(tp['cs_ast']))
         
         tree=[]
         for token in tp['java_ast'][0]:
             tree.append(java_language.convert_back[token.item()])
         print('\source tree {}'.format(' '.join(tree)))
         tree = []
         for token in tp['cs_ast'][0]:
             tree.append(cs_language.convert_back[token.item()])
         print('target tree {}'.format(' '.join(tree)))
         
         output_words = evaluate(model, tp)
         output_sentence = ' '.join(output_words)
         print('\tpredicted {}'.format(output_sentence))
         print(len(output_sentence))

print("Training...")
print("\tCreating initial models...")
encoder = Encoder(dim_input=java_language.count, dim_hidden=HIDDEN_SIZE, dim_embed=java_language.max_length)
decoder = Decoder(dim_output=cs_language.count, dim_hidden=HIDDEN_SIZE, dim_embed=java_language.max_length)
model = Seq2Seq(encoder, decoder, device).to(device)
print("\tStarting training...")
model = trainModel(model, train, EPOCHS)
print("Testing...")
evaluateRandomly(model, test, 1)








