"""
Jael Butler and Justin Hale
CS 4860
"""
from functools import reduce

import torch
import ast_parser as ap
from encoder import EncoderModel
from os.path import join, realpath
import numpy as np
from pprint import pprint
from languages import ASTNumbering as ast
from decoder import DecoderModel

print(torch.cuda.is_available())

epochs = 5
batch_size = 5
paths = ['antlr-data.json', 'itext-data.json', 'jgit-data.json', 'jts-data.json', 'lucene-data.json', 'poi-data.json']
paths = [join(realpath('.'), 'java2c#', name) for name in paths]

# Get all trees.
results = [ap.parseAST(path) for path in paths]
all_results = reduce(lambda x, y: x + y, results)

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

with open('temp.txt', 'w') as ofile:
    e = EncoderModel(24720, 24720, 1)
    h = e.initialize_hidden()
    e.train()
    output = []
    for i in range(epochs):
        train = [java_language.create_vector(x['java_ast']) for
                   x in np.random.choice(all_results, size=batch_size)]
        for vector in train:
            for point in vector:
                o, h = e.forward(point, h)
                output.append(o)

    e.eval()
    test_vector = test[0]

    decoder = DecoderModel(output[0].size(0), 100)
    dh = h
    doutput = []
    do, dh = decoder.forward(output[0], h)
    doutput.append(do)
    pprint(doutput, ofile)

exit(0)
