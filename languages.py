from collections import namedtuple
import codecs
import torch
import sys

Encoding = namedtuple('Encoding', 'index type')

#model transformer
class ASTNumbering:
    def __init__(self, name):
        self.name = name
        self.words = {}
        self.wordcount = {}
        self.convert_back = {0: 'Start', 1: 'End', 2: 'UnicodeCharacter', 3: '{', 4: '}', 5: '[', 6: ']'}
        self.count = 7
        self.max_length = 0

    def add_ast(self, tree):
        count = 0
        if isinstance(tree, list):
            count += 2
            for element in tree:
                count += self.add_ast(element)
        elif isinstance(tree, dict):
            count += 2
            for key in tree.keys():
                if key not in self.words:
                    self.words[key] = Encoding(self.count, type(tree[key]))
                    self.wordcount[key] = 1
                    self.convert_back[self.count] = key
                    self.count += 1
                    count += self.add_ast(tree[key])
                else:
                    self.wordcount[key] += 1
                    count += self.add_ast(tree[key])
        elif isinstance(tree, str):
            index = self.count
            count += 1
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
        if count > self.max_length:
            self.max_length = count
        return count

    def create_vector(self, tree):
        return_tree = [0]
        self._get_elements(tree, return_tree)
        return_tree.append(1)
        length = len(return_tree)
        if(length > self.max_length):
            raise ArgumentError('{} > {}'.format(length, self.max_length)
        final_tree = torch.zeros(1, self.max_length)
        for i in range(length):
            final_tree[0, i] = return_tree[i]
        return final_tree, length
        

    def _get_elements(self, tree, final_list):
        if isinstance(tree, list):
            final_list.append(5)
            for subtree in tree:
                self._get_elements(subtree, final_list)
            final_list.append(6)
        elif isinstance(tree, dict):
            final_list.append(3)
            for key in tree.keys():
                if key in self.words:
                    final_list.append(self.words[key][0])
                else:
                    final_list.append(2)
                self._get_elements(tree[key], final_list)
            final_list.append(4)
        elif isinstance(tree, str):
            if tree in self.words:
                final_list.append(self.words[tree][0])
            else:
                final_list.append(2)
        else:
            print(type(tree), file=sys.stderr)
