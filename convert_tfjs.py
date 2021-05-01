import os
import shutil
import argparse
import tensorflowjs
from models.yolov3 import YOLOv3
from utils.general import get_weights_type, get_classes_from_file
from utils.logging import format_seconds
from common.enums import ModelPurpose, WeightType
from common.constants import SUPPORTED_IMAGE_EXTENSIONS

parser = argparse.ArgumentParser(description='Converts a TensorFlow model to TensorFlowJS Layers model.')
parser.add_argument('-w', '--weights', required=True, help='Path to model weights. Can be either Darknet weights or a TensorFlow checkpoint.')
parser.add_argument('-c', '--classes', required=True, help='Path to file with a list of class names.')
parser.add_argument('-o', '--output', required=True, help='Path to where the converted TFJS model will be saved.')
parser.add_argument('--weights_type', choices=list(map(lambda x: x.name, WeightType)), help='Type of the provided weights. The script will try to infer the type automatically but you can still set it explicitly.')
parser.add_argument('--verbose', action='store_true', help='Log progress.')

def main(args):
    args.weights_type = get_weights_type(args.weights_type, args.weights)
    if args.verbose: print(f'Weights type: {args.weights_type.name}')

    classes = get_classes_from_file(args.classes)
    if args.verbose: print(f'Class names: {classes}')

    if args.verbose: print(f'Clearing output directory ({args.output})...')
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.mkdir(args.output)

    model = YOLOv3(len(classes))
    model.load_weights(args.weights, args.weights_type)

    if args.verbose: print('Converting the model...', end=' ')
    tensorflowjs.converters.save_keras_model(model.keras_model, args.output)

    if args.verbose: print('Done!')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
