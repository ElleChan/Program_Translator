import ast_parser as ap
import torch
from torch import nn
import torchvision
from functools import reduce


class AstModel:
    def __init__(self, paths):
        if isinstance(paths, str):
            self.data = ap.parseAST(paths)
        else:
            self.data = [ap.parseAST(x) for x in paths]
            self.data = reduce(lambda x, y: x + y, self.data)

    def train(self):
        pass

    def predict(self, data):
        pass

    def _prepare_data(self):
        pass