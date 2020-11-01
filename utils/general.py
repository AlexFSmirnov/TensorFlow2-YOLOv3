from common.enums import WeightType

def get_classes_from_file(path):
    with open(path, 'r') as fin:
        return list(map(lambda name: name.strip(), fin.readlines()))

def get_weights_type(input_type, weights_path):
    if not input_type:
        if weights_path.split('.')[-1] == 'weights':
            return WeightType.darknet
        return WeightType.checkpoint
    return WeightType[input_type]
