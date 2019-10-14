'''
Jael Butler and Justin Hale
CS 4860
'''

import torch
import ast_parser as ap


print(torch.rand(3, 5))

# Test ast_parser
ap.parseAST('./java2c#/antlr-data.json')
