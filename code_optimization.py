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
results = [ap.parseAST(path) for path in paths]
all_results = reduce(lambda x, y: x + y, results)
print(len(all_results))

# Generate datasets.
java_language = ast('Java')
cs_language = ast('CS')
for tree in all_results:
    cs_language.add_ast(tree['cs_ast'])
    java_language.add_ast(tree['java_ast'])

# Split training and test sets.
test = np.random.choice(all_results, size=1000, replace=False)
for item in test:
    all_results.remove(item)
print(len(all_results))
print(len(test))

# Train and evaluate.
with open('temp.txt', 'w') as ofile:
    e = EncoderModel(1, 1, 1)
    h = e.initialize_hidden()
    objective_func = nn.NLLLoss()       # Negative Log Likelihood Loss.
    e.train()

    # Train.
    for i in range(epochs):
        train = [java_language.create_vector(x['java_ast']) for
                   x in np.random.choice(all_results, size=batch_size)]     # Create subset of training set for actual training.
       # print(train)

        # Train encoder.
        output_e = []
        for vector in train:
            output = []
            for point in vector:
                o, h = e.forward(point, h)
                output.append(o)
            if len(output) > 1:
                output_e.append(output)
            else:
                output_e.append(o)
            
        # Train decoder.
        

    e.eval()
    test_vector = output_e[0]
    decoder = DecoderModel(1, 1)
    dh = h
    output_d = []
    decoder.train()
    for point in test_vector:
        print(point.dim())
        while point.dim() > 3:
            point = point.item()
        print(point)
        o, dh = decoder.forward(point, dh)
        output_d.append(o)
    print(output_d)
    decoder.eval()



vector = java_language.create_vector(all_results[0]['java_ast'])

exit(0)
