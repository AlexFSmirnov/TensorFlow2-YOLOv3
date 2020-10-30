import os

def get_classes_from_dataset(dataset_dir):
    with open(os.path.join(dataset_dir, 'names.txt'), 'r') as fin:
        return list(map(lambda name: name.strip(), fin.readlines()))
