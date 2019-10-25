"""
Jael Butler and Justin Hale
CS 4860
"""
from functools import reduce

import torch
import ast_parser as ap
from os.path import join, realpath
from pprint import pprint
from encoder import ASTNumbering as ast
import unicodedata
import json
import encodings
import sys


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

vector = java_language.create_vector(all_results[0]['java_ast'])

for key in vector:
    print(java_language.convert_back[key])

exit(0)
