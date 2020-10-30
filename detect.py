import os
import shutil
import argparse
from time import time 
from models.yolov3 import YOLOv3
from utils.general import get_classes_from_file
from common.enums import ModelPurpose, WeightType
from common.constants import SUPPORTED_IMAGE_EXTENSIONS

parser = argparse.ArgumentParser(description='Detects objects in images using a model with provided weights.')
parser.add_argument('-i', '--input', required=True, help='Either path to an image or to a directory with images.')
parser.add_argument('-o', '--output', help='Path to the directory where images with detections will be saved.')
parser.add_argument('-w', '--weights', required=True, help='Path to model weights. Can be either Darknet weights or a TensorFlow checkpoint.')
parser.add_argument('-c', '--classes', required=True, help='Path to file with a list of class names.')
parser.add_argument('--weights_type', choices=list(map(lambda x: x.name, WeightType)), help='Type of the provided weights. The script will try to infer the type automatically but you can still set it explicitly.')
parser.add_argument('--show', action='store_true', help='Show detection results in a window.')
parser.add_argument('--score_threshold', default=0.4, type=float, help='Confidence threshold for deciding whether detection is valid.')
parser.add_argument('--iou_threshold', default=0.45, type=float, help='IOU threshold for NMS.')
parser.add_argument('--show_label', action='store_true', help='Show labels for each bounding box in the output.')
parser.add_argument('--show_confidence', action='store_true', help='Show confidence next to each bounding box label.')
parser.add_argument('--verbose', action='store_true', help='Log progress.')

def get_weights_type(input_type, weights_path):
    if not input_type:
        if weights_path.split('.')[-1] == 'weights':
            return WeightType.darknet
        return WeightType.checkpoint
    return WeightType[input_type]

def main(args):
    args.weights_type = get_weights_type(args.weights_type, args.weights)
    if args.verbose: print(f'Weights type: {args.weights_type.name}')

    classes = get_classes_from_file(args.classes)
    if args.verbose: print(f'Class names: {classes}')

    if args.output:
        if args.verbose: print(f'Clearing output directory ({args.output})...')
        if os.path.exists(args.output):
            shutil.rmtree(args.output)
        os.mkdir(args.output)

    model = YOLOv3(len(classes), purpose=ModelPurpose.detection)
    model.load_weights(args.weights, args.weights_type)

    def detect_in_image(dir, filename):
        if filename.split('.')[-1] not in SUPPORTED_IMAGE_EXTENSIONS:
            return

        model.detect_image(
            os.path.join(dir, filename),
            classes,
            os.path.join(args.output, f'detected_{filename}') if args.output else None,
            show=args.show,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            show_label=args.show_label,
            show_confidence=args.show_confidence,
        )

    if os.path.isfile(args.input):
        dir, filename = os.path.split(args.input)
        detect_in_image(dir, filename)
    elif os.path.isdir(args.input):
        filenames = os.listdir(args.input)
        files_count = len(filenames)
        global_start = time()
        for i, filename in enumerate(filenames):
            d_start = time()
            detect_in_image(args.input, filename)
            d_end = time()

            if args.verbose:
                elapsed_time = time() - global_start
                eta = format_seconds(elapsed_time / (i + 1) * (files_count - i - 1))
                print(f'[{i + 1:0{len(str(files_count))}} / {files_count}] ({d_end - d_start:.2f}s) ETA: {eta}')

        if args.verbose:
            print(f'Done in {format_seconds(time() - global_start)}')

def format_seconds(s):
    hours, rem = divmod(int(s), 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02}h {minutes:02}m {seconds:02}s'

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
