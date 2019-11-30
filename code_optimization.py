"""
Jael Butler and Justin Hale
CS 4860
"""
from functools import reduce

import torch
from torch import nn
import ast_parser as ap
from encoder import EncoderModel
from os.path import join, realpath
import numpy as np
from pprint import pprint
from decoder import DecoderModel
from languages import ASTNumbering as ast


epochs = 1
batch_size = 1

paths = ['antlr-data.json', 'itext-data.json', 'jgit-data.json', 'jts-data.json', 'lucene-data.json', 'poi-data.json']
paths = [join(realpath('.'), 'java2c#', name) for name in paths]

# Get all trees.
print("Getting all the trees")
results = [ap.parseAST(path) for path in paths]
all_results = reduce(lambda x, y: x + y, results)

# Generate datasets.
print("Creating tree classes...")
java_language = ast('Java')
cs_language = ast('CS')
for tree in all_results:
    cs_language.add_ast(tree['cs_ast'])
    java_language.add_ast(tree['java_ast'])

# Create numerical input vectors.
print("Converting trees to numerical vectors...")
dataset = []
for tree in all_results:
	cs_tree, cs_length = cs_language.create_vector(tree['cs_ast'])
	java_tree, java_length = java_langage.create_vector(tree['java_ast'])
	print(cs_tree, java_tree)
	exit()

# Split training and test sets.
print("Shuffling and splitting dataset for training and testing.")
np.random.shuffle(all_results)
train, test = np.split(all_results, np.array([1000]))

#test = np.random.choice(all_results, size=1000, replace=False)
#for item in test:
#    all_results.remove(item)
#all_results = np.array(all_results)

# Train and evaluate.
with open('temp.txt', 'w') as ofile:
    e = EncoderModel(java_language.max_length, 10, java_language.count)			# Create encoder object.
    h = e.initialize_hidden()								# Create initial hidden input.
    objective_func = nn.NLLLoss()       						# Negative Log Likelihood Loss.
    
    # Start training phase.
    e.train()
    output_e = []
    for i in range(epochs):
        train = [java_language.create_vector(x['java_ast']) for
                   x in np.random.choice(all_results, size=batch_size)]     		# Create subset of training set for actual training.
        lengths = torch.Tensor([x[1] for x in train])
    
	# Padding input vectors.
        input_vector = torch.zeros(batch_size, java_language.max_length, dtype=torch.long) # Create a tensor that can hold all the values
        for i in range(batch_size):
            input_vector[i, :len(train[i][0])] = train[i][0]
        # Train encoder.
        output = []
        o, h = e.forward(input_vector, h, lengths)
        output.append(o)
        if len(output) > 1:
            output_e.append(output)
        else:
            output_e.append(o)
            
        # Train decoder.
        
    print(output_e)
    e.eval()

    #Train decoder using input from the encoder
    test_vector = output_e
    decoder = DecoderModel(java_language.count, java_language.count)
    decoder_hidden = h
    output_d = []
    decoder.train()
    for vector in output_e:
        o, decoder_hidden = decoder.forward(vector, decoder_hidden, lengths)
        output_d.append(o)
    print(output_d)
    decoder.eval()



vector = java_language.create_vector(all_results[0]['java_ast'])

exit(0)
