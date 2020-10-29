import os
import shutil
import argparse
import xml.etree.ElementTree as ET
from random import shuffle

parser = argparse.ArgumentParser(description='Given a dataset annotated with Pascal VOC, split it to train and test sets and transform annotations to YOLO format.')
parser.add_argument('-i', '--images_dir', required=True, help='Path to directory with images.')
parser.add_argument('-a', '--annotations_dir', help='Path to directory with .xml annotations for images. Defaults to the value of images_dir.')
parser.add_argument('-o', '--output_dir', required=True, help='Path to where the result will be saved')
parser.add_argument('-s', '--train_test_split', default=0.8, type=float, help='Defines the ratio between training and test data: train = total * train_test_split; test = total * (1 - train_test_split).')
parser.add_argument('--verbose', action='store_true', help='Show progress.')

SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif']

class_names = []

def getBoxesFromXML(xml_path):
    global class_names

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = ''
    for obj in root.iter('object'):
        class_name = obj.find('name').text
        if class_name not in class_names:
            class_names.append(class_name)

        class_id = class_names.index(class_name)
        xmlbox = obj.find('bndbox')
        coords = ','.join(map(lambda coord: str(int(float(xmlbox.find(coord).text))), ['xmin', 'ymin', 'xmax', 'ymax']))
        boxes += f'{coords},{class_id} '

    return boxes.rstrip()

def main(args):
    if not args.annotations_dir:
        args.annotations_dir = args.images_dir

    # Filtering out images with unsupported extensions
    valid_images = list(filter(
        lambda filename: filename.split('.')[-1] in SUPPORTED_EXTENSIONS,
        os.listdir(args.images_dir)
    ))

    if args.verbose: print(f'Keeping {len(valid_images)} out of {len(os.listdir(args.images_dir))} images based on supported extensions. \n')

    # Randomly splitting to train and test sets based on the provided ratio
    shuffle(valid_images)
    train_images = valid_images[:int(len(valid_images) * args.train_test_split)]
    test_images = valid_images[int(len(valid_images) * args.train_test_split):]

    if args.verbose: print(f'Train set: {len(train_images)} images. \n Test set: {len(test_images)} images. \n')

    # For each set (train/test) go through all images, copy them to respective output folders and create a YOLO annotations file
    for subset, images in [['train', train_images], ['test', test_images]]:
        if args.verbose: print(f'Processing {subset}... ', end='')

        output_images_dir = os.path.join(args.output_dir, subset)
        if os.path.exists(output_images_dir):
            shutil.rmtree(output_images_dir)
        os.mkdir(output_images_dir)

        yolo_annotations = ''
        for image_filename in images:
            image_input_path = os.path.join(args.images_dir, image_filename)
            image_output_path = os.path.join(args.output_dir, subset, image_filename)
            shutil.copy(image_input_path, image_output_path)

            xml_filename = f'{image_filename.split(".")[0]}.xml'
            xml_path = os.path.join(args.annotations_dir, xml_filename)
            if os.path.exists(xml_path):
                boxes = getBoxesFromXML(xml_path)
                yolo_annotations += f'{image_output_path} {boxes}\n'

        with open(os.path.join(args.output_dir, f'{subset}_annotations.txt'), 'w') as fout:
            fout.write(yolo_annotations)

        if args.verbose: print('Done!')

    # Based on processed annotations, write a list of class names
    with open(os.path.join(args.output_dir, 'names.txt'), 'w') as fout:
        fout.write('\n'.join(class_names))

    if args.verbose: print(f'\nDetected class names: {class_names}')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
