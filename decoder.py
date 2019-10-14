import json
from collections import namedtuple
from os.path import join, dirname, realpath

Trees = namedtuple('Trees', 'java cs')


def decoder(path):
    with open(path) as ifile:
        return [Trees(x['java_ast'], x['cs_ast']) for x in json.load(ifile)]



if __name__ == '__main__':
    home = dirname(realpath(__file__))
    results = decoder(join(home, 'tree2tree_data', 'java2c#', 'antlr-data.json'))
    for value in results:
        java, cs = value
        print(java)
        print(cs)
