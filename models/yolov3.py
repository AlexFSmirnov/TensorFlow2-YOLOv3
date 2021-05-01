import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
from time import time
from utils.model_creation import create_keras_model
from utils.dataset import Dataset
from utils.training import compute_loss
from utils.detection import postprocess_boxes, nms, draw_bbox
from utils.image import preprocess_image
from utils.logging import format_seconds
from common.enums import ModelPurpose, WeightType, DatasetType, CheckpointSaveMode

class YOLOv3:
    def __init__(self, num_classes, purpose=ModelPurpose.detection, input_size=416, channels=3, tiny=False):
        self.num_classes = num_classes
        self.purpose = purpose
        self.input_size = input_size
        self.channels = channels
        self.tiny = tiny

        self.keras_model = create_keras_model(
            self.num_classes,
            self.input_size,
            self.channels,
            self.tiny,
            is_for_training=(self.purpose == ModelPurpose.training),
        )

    def load_weights(self, path, type):
        if type == WeightType.darknet:
            self.load_darknet_weights(path)
        elif type == WeightType.checkpoint:
            final_path = path
            if '.index' in final_path:
                final_path = final_path[:final_path.index('.index')]
            if '.data-' in final_path:
                final_path = final_path[:final_path.index('.data-')]

            self.keras_model.load_weights(final_path)
        else:
            raise Exception('Weights type not provided.')

    def load_darknet_weights(self, path):
        # Reset layer names
        tf.keras.backend.clear_session()

        range1 = 13 if self.tiny else 75
        range2 = [9, 12] if self.tiny else [58, 66, 74]

        with open(path, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(range1):
                if i > 0:
                    conv_layer_name = 'conv2d_%d' %i
                else:
                    conv_layer_name = 'conv2d'
                    
                if j > 0:
                    bn_layer_name = 'batch_normalization_%d' %j
                else:
                    bn_layer_name = 'batch_normalization'
                
                conv_layer = self.keras_model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = self.keras_model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            assert len(wf.read()) == 0, 'failed to read all data'

    def transfer_weights_from(self, yolo_model):
        for i, layer in enumerate(yolo_model.keras_model.layers):
            layer_weights = layer.get_weights()
            if layer_weights != []:
                try:
                    self.keras_model.layers[i].set_weights(layer_weights)
                except:
                    continue

    def train(
        self,
        dataset_dir,
        output_dir,
        epochs,
        name='yolov3_custom',
        checkpoint_save_mode=CheckpointSaveMode.best,
        augment_data=True,
        batch_size=4,
        warmup_epochs=2,
        learning_rate_init=1e-6,
        learning_rate_end=1e-4,
        load_images_to_ram=True,
        log_dir=None,
        verbose=True,
    ):
        if self.purpose != ModelPurpose.training:
            raise Exception(f'Model was not created with purpose "{self.purpose.name}". Create it with purpose "{ModelPurpose.training.name}" instead.')

        train_writer = None
        validate_writer = None
        if log_dir is not None:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            train_writer = tf.summary.create_file_writer(log_dir)
            validate_writer = tf.summary.create_file_writer(log_dir)

        trainset = Dataset(dataset_dir, DatasetType.train, augment_data=augment_data, batch_size=batch_size, load_images_to_ram=load_images_to_ram)
        testset = Dataset(dataset_dir, DatasetType.test, augment_data=augment_data, batch_size=batch_size, load_images_to_ram=load_images_to_ram)

        steps_per_epoch = len(trainset)
        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = epochs * steps_per_epoch

        optimizer = tf.keras.optimizers.Adam()

        def train_step(image_data, target):
            with tf.GradientTape() as tape:
                pred_result = self.keras_model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                grid = 2 if self.tiny else 3
                for i in range(grid):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = compute_loss(pred, conv, *target[i], i, num_classes=self.num_classes)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                total_loss = giou_loss + conf_loss + prob_loss

                gradients = tape.gradient(total_loss, self.keras_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.keras_model.trainable_variables))

                # update learning rate
                # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
                global_steps.assign_add(1)
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * learning_rate_end
                else:
                    lr = learning_rate_init + 0.5 * (learning_rate_end - learning_rate_init) * (
                        (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                    )
                optimizer.lr.assign(lr.numpy())

                # writing summary data
                if train_writer is not None:
                    with train_writer.as_default():
                        tf.summary.scalar("learning_rate", optimizer.lr, step=global_steps)
                        tf.summary.scalar("train_loss/total_loss", total_loss, step=global_steps)
                        tf.summary.scalar("train_loss/giou_loss", giou_loss, step=global_steps)
                        tf.summary.scalar("train_loss/conf_loss", conf_loss, step=global_steps)
                        tf.summary.scalar("train_loss/prob_loss", prob_loss, step=global_steps)
                    train_writer.flush()
                
            return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

        def validate_step(image_data, target):
            with tf.GradientTape() as tape:
                pred_result = self.keras_model(image_data, training=False)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                grid = 2 if self.tiny else 3
                for i in range(grid):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = compute_loss(pred, conv, *target[i], i, num_classes=self.num_classes)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                total_loss = giou_loss + conf_loss + prob_loss
                
            return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

        train_start_time = time()
        best_val_loss = 1000 # should be large at start
        for epoch in range(epochs):
            for image_data, target in trainset:
                results = train_step(image_data, target)
                cur_step = results[0] % steps_per_epoch

                if verbose:
                    cur_global_step = epoch * steps_per_epoch + cur_step
                    elapsed_time = time() - train_start_time
                    eta = format_seconds(elapsed_time / (cur_global_step + 1) * (total_steps - cur_global_step - 1))
                    print(f'\nEpoch: {epoch + 1: {len(str(epochs))}}/{epochs}, step: {cur_step + 1: {len(str(steps_per_epoch))}}/{steps_per_epoch}, ETA: {eta}')
                    print(f'lr: {results[1]:.6f},    giou_loss: {results[2]:7.2f},    conf_loss: {results[3]:7.2f},    prob_loss: {results[4]:7.2f},    total_loss: {results[5]:7.2f}')

            if len(testset) == 0:
                print("Test set is empty, can't validate model.")
                self.keras_model.save_weights(os.path.join(output_dir, name))
                continue
            
            count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
            for image_data, target in testset:
                results = validate_step(image_data, target)
                count += 1
                giou_val += results[0]
                conf_val += results[1]
                prob_val += results[2]
                total_val += results[3]

            # writing validate summary data
            if validate_writer is not None:
                with validate_writer.as_default():
                    tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
                    tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
                    tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
                    tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
                validate_writer.flush()
                
            if verbose:
                print('\n\nValidation loss:')
                print(f'giou_val_loss: {giou_val / count:7.2f},    conf_val_loss: {conf_val / count:7.2f},    prob_val_loss: {prob_val / count:7.2f},    total_val_loss: {total_val / count:7.2f}\n\n')

            save_path = ''
            if checkpoint_save_mode == CheckpointSaveMode.all:
                save_path = os.path.join(output_dir, f'{name}_val_loss_{total_val / count:7.2f}')
                self.keras_model.save_weights(save_path)
            elif checkpoint_save_mode == CheckpointSaveMode.best:
                if best_val_loss > total_val / count:
                    save_path = os.path.join(output_dir, name)
                    self.keras_model.save_weights(save_path)
                    best_val_loss = total_val / count
            elif checkpoint_save_mode == CheckpointSaveMode.last:
                save_path = os.path.join(output_dir, name)
                self.keras_model.save_weights(save_path)


    def detect(self, image_path, classes, output_path=None, score_threshold=0.4, iou_threshold=0.45, show=False, show_label=True, show_confidence=True):
        if self.purpose != ModelPurpose.detection:
            raise Exception(f'Model was not created with purpose "{self.purpose.name}". Create it with purpose "{ModelPurpose.detection.name}" instead.')

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = preprocess_image(np.copy(original_image), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.keras_model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, self.input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        image = draw_bbox(original_image, bboxes, classes=classes, show_label=show_label, show_confidence=show_confidence)

        if output_path:
            cv2.imwrite(output_path, image)

        if show:
            # Show the image
            cv2.imshow("predicted image", image)
            # Load and hold the image
            cv2.waitKey(0)
            # To close the window after the required kill value was provided
            cv2.destroyAllWindows()
            
        return [bboxes, image]
