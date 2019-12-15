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
print("\tTotal of", len(all_results), "trees.")

# Generate datasets.
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


# Split training and test sets.
print("Shuffling and splitting dataset for training and testing.")
np.random.shuffle(dataset)
train, test = np.split(dataset, np.array([-1000]))
print("\tLength of training set is:", len(train))
print("\tLength of testing set is:", len(test))


# Train and evaluate.
print("Training Encoder-Decoder")
with open('temp.txt', 'w') as ofile:
    e = EncoderModel(java_language.max_length, 10, java_language.count)			# Create encoder object.
    d = DecoderModel(java_language.count, 10, cs_language.count)			# Create decoder object. THE HIDDEN SIZES BETTER BE THE SAME!!!
    h = e.initialize_hidden()								# Create initial hidden input.
    objective_func = nn.NLLLoss()       						# Negative Log Likelihood Loss.
    
    # Start training phase.
    e.train()
    d.train()
   
    for i in range(epochs):
        print("\tEpoch:", i)
        batch = np.random.choice(train, size=batch_size) 		    		# Create subset of training set for actual training.
	
        # Train encoder.
        outputs = []
        for data_point in batch:
             input_vector = data_point['java_ast']
             input_vector = input_vector.long()
             #h = h.long()
             print("\t\tEncoder Datapoint:", input_vector)
             o, h = e.forward(input_vector, h, java_language.max_length)
             print("\t\t\t Output vector and size", o, "(", o.size(), ")")
             print("\t\t\t Hidden vector and size", h, "(", h.size(), ")")
             outputs.append(o)

        # Train decoder.
        for output_vector in outputs:
            input_vector = output_vector
            #input_vector = output_vector[0]		# Ignore the gradient in the tuple
            #input_vector = input_vector.long()
            #h = h.long()
            print("\t\tDecoder Datapoint:", input_vector)
            o, h = d.forward(input_vector, h, cs_language.max_length)
            print ("\t\t\tOutput vector and size", o, "(", o.size(), ")")
            print ("\t\t\tHidden vector and size", h, "(", h.size(), ")")	
 
    exit()    


    print("Testing Encoder-Decoder")
    e.eval()



exit(0)
