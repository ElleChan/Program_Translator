from collections import namedtuple
import codecs
import torch
import sys

Encoding = namedtuple('Encoding', 'index type')

# Referenced: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class ASTNumbering:
    def __init__(self, name):
        self.name = name
        self.words = {}
        self.wordcount = {}
        self.convert_back = {0: 'Start', 1: 'End', 2: 'UnicodeCharacter'}
        self.count = 3

    def add_ast(self, tree):
        if isinstance(tree, list):
            for element in tree:
                self.add_ast(element)
        elif isinstance(tree, dict):
            for key in tree.keys():
                if key not in self.words:
                    self.words[key] = Encoding(self.count, type(tree[key]))
                    self.wordcount[key] = 1
                    self.convert_back[self.count] = key
                    self.count += 1
                    self.add_ast(tree[key])
                else:
                    self.wordcount[key] += 1
                    self.add_ast(tree[key])
        elif isinstance(tree, str):
            index = self.count
            try:
                codecs.charmap_encode(tree)
            except UnicodeEncodeError as err:
                tree = 'UnicodeCharacter'
                index = 3
            if tree not in self.words:
                self.words[tree] = Encoding(index, None)
                self.wordcount[tree] = 1
                self.convert_back[index] = tree
                self.count += 1 if index == self.count else 0
            else:
                self.wordcount[tree] += 1
        else:
            print(type(tree))

    def create_vector(self, tree):
        return_tree = [0]
        self._get_elements(tree, return_tree)
        return_tree.append(1)
        return torch.tensor(return_tree)

    def _get_elements(self, tree, final_list):
        if isinstance(tree, list):
            for subtree in tree:
                self._get_elements(subtree, final_list)
        elif isinstance(tree, dict):
            for key in tree.keys():
                if key in self.words:
                    final_list.append(self.words[key][0])
                else:
                    final_list.append(2)
                self._get_elements(tree[key], final_list)

        elif isinstance(tree, str):
            if tree in self.words:
                final_list.append(self.words[tree][0])
            else:
                final_list.append(2)
        else:
            print(type(tree), file=sys.stderr)
