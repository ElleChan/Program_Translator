'''
This module contains all of the relevant methods required to parse an AST from
JSON format to a Python datatype.
'''

import json
from pprint import pprint


# Takes a path to a JSON file, and reads its contents into a list of Python dictionaries.
def parseAST(path):
    handle = open(path)
    json_string = handle.read()
    handle.close()

    tree_list = json.loads(json_string)
    return tree_list
