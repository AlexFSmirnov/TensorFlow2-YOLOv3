import os
import argparse
from models.yolov3 import YOLOv3
from utils.general import get_weights_type, get_classes_from_file
from utils.cli import str2bool
from common.enums import ModelPurpose, WeightType, CheckpointSaveMode

parser = argparse.ArgumentParser(description='Detects objects in images using a model with provided weights.')
parser.add_argument('-d', '--dataset_dir', required=True, help='Path to the directory with images and YOLO annotations.')
parser.add_argument('-o', '--output_dir', required=True, help='Path to the directory where the training checkpoints will be saved.')
parser.add_argument('-e', '--epochs', required=True, type=int, help='Number of times the model will train on the given dataset.')
parser.add_argument('-w', '--pre_trained_weights', help='Path to pre-trained model weights. Can be either Darknet weights or a TensorFlow checkpoint.')
parser.add_argument('--pre_trained_classes_count', default=80, help='Number of classes used in the provided pre-trained-weights. Only needed if pre_trained_weights type is Darknet.')
parser.add_argument('--pre_trained_weights_type', choices=list(map(lambda x: x.name, WeightType)), help='Type of the provided pre-trained weights. The script will try to infer the type automatically but you can still set it explicitly.')
parser.add_argument('--name', help='Name of the model. Used for saving checkpoints. Defaults to the name of the output dir.')
parser.add_argument('--checkpoint_save_mode', default=CheckpointSaveMode.best.name, choices=list(map(lambda x: x.name, CheckpointSaveMode)), help='Defines how training checkpoints will be saved. All - a new checkpoint will be saved every epoch. Best - only the checkpoint from the epoch with lowest loss will be saved. Last - the checkpoint from the last epoch will be saved.')
parser.add_argument('--augment_data', default=True, const=True, nargs='?', type=str2bool, help='If true, every epoch some images will be randomly augmented (mirrored, translated, etc.)')
parser.add_argument('--batch_size', default=4, type=int, help='How many images to process in one batch.')
parser.add_argument('--warmup_epochs', default=2, type=int, help='During warmup epochs learning rate will gradually increase from learning_rate_init to learning_rate_end.')
parser.add_argument('--learning_rate_init', default=1e-6, type=float, help='Initial (minimal) learning rate.')
parser.add_argument('--learning_rate_end', default=1e-4, type=float, help='Final (maximal) learning rate.')
parser.add_argument('--load_images_to_ram', default=True, const=True, nargs='?', type=str2bool, help='If true, training will be faster, but will require more RAM.')
parser.add_argument('--logdir', help='Path to logs directory (for Tensorboard and such).')
parser.add_argument('--verbose', action='store_true', help='Log progress.')

def main(args):
    if not args.name:
        args.name = os.path.basename(os.path.normpath(args.output_dir))

    classes = get_classes_from_file(os.path.join(args.dataset_dir, 'names.txt'))
    if args.verbose: print(f'Class names: {classes}')

    if args.pre_trained_weights:
        # Inferring the type of pre-trained weights
        args.pre_trained_weights_type = get_weights_type(args.pre_trained_weights_type, args.pre_trained_weights)
        if args.verbose: print(f'Weights type: {args.pre_trained_weights_type.name}')

        # If darknet, create a new model with provided weights
        if args.pre_trained_weights_type == WeightType.darknet:
            pre_trained_model = YOLOv3(args.pre_trained_classes_count, purpose=ModelPurpose.transfer)
            pre_trained_model.load_weights(args.pre_trained_weights, WeightType.darknet)
            if args.verbose: print('Created a pre_trained_model with the provided weights')

    # Create a new model for training. We have to create it after we (potentially) load darknet weights into
    # a pre-trained model. Otherwise we would have 2 keras models while loading weights, and in that case
    # keras can change layer names so the loading code won't work.
    model = YOLOv3(len(classes), purpose=ModelPurpose.training)

    # If pre-trained weights were provided, load them into the model.
    if args.pre_trained_weights:
        if args.pre_trained_weights_type == WeightType.checkpoint:
            model.load_weights(args.pre_trained_weights, WeightType.checkpoint)
            if args.verbose: print('Loaded pre-trained weights from the checkpoint')
        elif args.pre_trained_weights_type == WeightType.darknet:
            model.transfer_weights_from(pre_trained_model)
            if args.verbose: print('Transfered weights from the pre_trained_model')

    # Start the training process.
    model.train(
        args.dataset_dir,
        args.output_dir,
        args.epochs,
        args.name,
        CheckpointSaveMode[args.checkpoint_save_mode],
        args.augment_data,
        args.batch_size,
        args.warmup_epochs,
        args.learning_rate_init,
        args.learning_rate_end,
        args.load_images_to_ram,
        args.logdir,
        args.verbose,
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)