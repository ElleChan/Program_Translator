import json
from os.path import join, dirname, realpath

def decoder(path):
    with open(path) as ifile:
        print(ifile)
        return json.load(ifile)


if __name__ == '__main__':
    home = dirname(realpath(__file__))
    results = decoder(join(home, 'tree2tree_data', 'java2c#', 'antlr-data.json'))
    print(results)
