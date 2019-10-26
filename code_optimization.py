"""
Jael Butler and Justin Hale
CS 4860
"""
from functools import reduce

import torch
import ast_parser as ap
from ast_model import EncoderModel
from os.path import join, realpath
import numpy as np
from pprint import pprint
from encoder import ASTNumbering as ast

epochs = 5
batch_size = 5

print(torch.rand(3, 5))

paths = ['antlr-data.json', 'itext-data.json', 'jgit-data.json', 'jts-data.json', 'lucene-data.json', 'poi-data.json']
paths = [join(realpath('.'), 'java2c#', name) for name in paths]
# Test ast_parser
results = [ap.parseAST(path) for path in paths]

all_results = reduce(lambda x, y: x + y, results)

print(len(all_results))

java_language = ast('Java')

cs_language = ast('CS')

for tree in all_results:
    cs_language.add_ast(tree['cs_ast'])
    java_language.add_ast(tree['java_ast'])

test = np.random.choice(all_results, size=1000, replace=False)
for item in test:
    all_results.remove(item)

print(len(all_results))

print(len(test))

e = EncoderModel(1000, 100, 0, 0)
h = e.initialize_hidden()
for i in range(epochs):
    train = [java_language.create_vector(x['java_ast']) for
             x in np.random.choice(all_results, size=batch_size)]
    pprint(train)

    output = []
    for point in train:
        o, h = e.forward(point, h)
        output.append(o)
    print(output)


vector = java_language.create_vector(all_results[0]['java_ast'])

exit(0)
