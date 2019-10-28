from languages import ASTNumbering
import numpy as np
from numpy.random import choice

class AstModel:
    def __init__(self, paths):
        if isinstance(paths, str):
            self.data = ap.parseAST(paths)
        elif isinstance(paths, list):
            self.data = [ap.parseAST(x) for x in paths]
            self.data = reduce(lambda x, y: x + y, self.data)
        else:
            raise TypeError('AstModel() expects a string or a list')
        self.java_language = ASTNumbering('Java')
        self.cs_language = ASTNumbering('C#')
        for tree in self.data:
            self.java_language.add_ast(tree['java_ast'])
            self.cs_language.add_ast(tree['cs_ast'])

    def train(self, train_data, epochs, batch_size):
        for _ in range(epochs):
                train = choice(train_data, size=batch_size)
                train = self._prepare_data(train)


    def predict(self, data):
        pass

    def _prepare_data(self, data):
        return [(self.java_language.create_vector(x['java_ast']),
                 self.cs_language.create_vector(x['cs_ast']))
                for x in data]
