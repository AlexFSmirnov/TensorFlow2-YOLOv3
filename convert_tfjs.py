import os
import shutil
import argparse
import subprocess
import tensorflowjs
import tensorflow as tf
from models.yolov3 import YOLOv3
from utils.general import get_weights_type, get_classes_from_file
from common.enums import WeightType, TFJSModelType

parser = argparse.ArgumentParser(description='Converts a TensorFlow model to TensorFlowJS Layers model.')
parser.add_argument('-w', '--weights', required=True, help='Path to model weights. Can be either Darknet weights or a TensorFlow checkpoint.')
parser.add_argument('-c', '--classes', required=True, help='Path to file with a list of class names.')
parser.add_argument('-o', '--output', required=True, help='Path to where the converted TFJS model will be saved.')
parser.add_argument('-t', '--output_type', default=TFJSModelType.graph.name, choices=list(map(lambda x: x.name, TFJSModelType)), help='The type of the TFJS model to be saved (graph / layers).')
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

    if args.output_type == TFJSModelType.layers.name:
        if args.verbose: print('Converting the model to the TFJS Layers Model...')
        tensorflowjs.converters.save_keras_model(model.keras_model, args.output)
    elif args.output_type == TFJSModelType.graph.name:
        if args.verbose: print('Saving a model as a TF Saved Model for futrher conversion...')
        saved_model_dir = f'{args.output}_temp_tf_saved_model'
        while True:
            if os.path.exists(saved_model_dir):
                saved_model_dir = f'{saved_model_dir}_'
            else:
                break
        os.mkdir(saved_model_dir)
        tf.keras.models.save_model(model.keras_model, saved_model_dir)
        if args.verbose: print('Done!')

        if args.verbose: print('Converting the TF Saved Model to a TFJS Graph Model (this will take a while)...')
        tensorflowjs.converters.convert_tf_saved_model(saved_model_dir, args.output)
        if args.verbose: print('Done!')

        if args.verbose: print('Removing TF Saved Model...')
        shutil.rmtree(saved_model_dir)

    if args.verbose: print('Done!')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
