'''
Jael Butler and Justin Hale
CS 4860
'''

import torch
import ast_parser as ap
from os.path import join, realpath


print(torch.rand(3, 5))

paths = ['antlr-data.json', 'itext-data.json', 'jgit-data.json', 'jts-data.json', 'lucene-data.json', 'poi-data.json']
paths = [join(realpath('.'), 'java2c#', name) for name in paths]
print(paths)

# Test ast_parser
results = ap.parseAST(paths[0])
#print(results)

# Split datasets

# Train NN

# Run test points.

