import os

def get_classes_from_file(path):
    with open(path, 'r') as fin:
        return list(map(lambda name: name.strip(), fin.readlines()))
