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
    e = EncoderModel(24721, 24721, 24721)
    h = e.initialize_hidden()
    objective_func = nn.NLLLoss()       # Negative Log Likelihood Loss.
    e.train()

    # Train.
    for i in range(epochs):
        train = [java_language.create_vector(x['java_ast']) for
                   x in np.random.choice(all_results, size=batch_size)]     # Create subset of training set for actual training.
        print(train)

        # Train encoder.
        output_e = []
        for vector in train:
            for point in vector:
                o, h = e.forward(point, h)
                output_e.append(o)
        print(output_e)
        
        # Train decoder.
        

    e.eval()
    test_vector = test[0]
    #eval_output =
    for point in java_language.create_vector(test_vector):
        o, h = e.forward(point, h)



vector = java_language.create_vector(all_results[0]['java_ast'])

exit(0)
