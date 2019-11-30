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
	java_tree, java_length = java_language.create_vector(tree['java_ast'])
	dataset.append({'cs_ast': cs_tree, 'java_ast': java_tree})

# Split training and test sets.
print("Shuffling and splitting dataset for training and testing.")
np.random.shuffle(dataset)
train, test = np.split(dataset, np.array([-1000]))

# Train and evaluate.
print("Training Encoder-Decoder")
with open('temp.txt', 'w') as ofile:
    e = EncoderModel(java_language.max_length, 10, java_language.count)			# Create encoder object.
    #d = DecoderModel()
    h = e.initialize_hidden()								# Create initial hidden input.
    objective_func = nn.NLLLoss()       						# Negative Log Likelihood Loss.
    
    # Start training phase.
    e.train()
   
    for i in range(epochs):
        print("\tEpoch:", i)
        batch = np.random.choice(train, size=batch_size) 		    		# Create subset of training set for actual training.
	
        # Train encoder.
        outputs = []
        for data_point in batch:
             input_vector = data_point['java_ast']
             input_vector = input_vector.long()
             o, h = e.forward(input_vector, h, java_language.max_length)
             print(input_vector, "->", o)
             
             outputs.append(o)

        # Train decoder.
	
 
    exit()    


    print("Testing Encoder-Decoder")
    e.eval()



exit(0)
