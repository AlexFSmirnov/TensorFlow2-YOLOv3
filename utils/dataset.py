import os
import cv2
import numpy as np
import tensorflow as tf
from common.constants import YOLO_STRIDES, YOLO_ANCHORS, YOLO_ANCHOR_PER_SCALE, YOLO_MAX_BBOX_PER_SCALE
from utils.bbox import bbox_iou
from utils.image import random_augmentations, preprocess_image
from utils.general import get_classes_from_file

class Dataset:
    def __init__(self, dataset_path, type, input_size=416, batch_size=4, augment_data=True, load_images_to_ram=True):
        self.annotations_path = os.path.join(dataset_path, f'{type.name}_annotations.txt')
        self.input_size = input_size
        self.batch_size = batch_size
        self.augment_data = augment_data
        self.load_images_to_ram = load_images_to_ram

        self.classes = get_classes_from_file(os.path.join(dataset_path, 'names.txt'))
        self.num_classes = len(self.classes)

        self.strides = np.array(YOLO_STRIDES)
        self.anchors = (np.array(YOLO_ANCHORS).T / self.strides).T

        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        final_annotations = []
        with open(self.annotations_path, 'r') as fin:
            annotations = [line.strip() for line in fin.readlines() if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        
        for annotation in annotations:
            # fully parse annotations
            line = annotation.split()
            image_path, index = '', 1
            for i, one_line in enumerate(line):
                if not one_line.replace(',', '').isnumeric():
                    if image_path != '':
                        image_path += ' '
                    image_path += one_line
                else:
                    index = i
                    break
            if not os.path.exists(image_path):
                raise KeyError(f'{image_path} does not exist')
            if self.load_images_to_ram:
                image = cv2.imread(image_path)
            else:
                image = ''
            final_annotations.append([image_path, line[index:], image])
        return final_annotations

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def delete_bad_annotation(self, bad_annotation):
        bad_image_name = bad_annotation[0].split('/')[-1] # can be used to delete bad image

        # remove bad annotation line from annotation file
        with open(self.annotations_path, "r+") as fin:
            lines = fin.readlines()
            fin.seek(0)
            for line in lines:
                if bad_image_name not in line:
                    fin.write(line)
            fin.truncate()
    
    def __next__(self):
        with tf.device('/cpu:0'):
            self.output_sizes = self.input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.output_sizes[0], self.output_sizes[0], self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.output_sizes[1], self.output_sizes[1], self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.output_sizes[2], self.output_sizes[2], self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            exceptions = False
            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    try:
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    except IndexError:
                        exceptions = True
                        self.delete_bad_annotation(annotation)
                        print(f'Something wrong with {annotation[0]}. Removed this line from annotation file.')
                        num += 1
                        continue

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1

                if exceptions: 
                    raise Exception('There were problems with this dataset. They were fixed automatically, please restart the training process.')

                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def parse_annotation(self, annotation, mAP = 'False'):
        if self.load_images_to_ram:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)
            
        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])

        if self.augment_data:
            image, bboxes = random_augmentations(image, bboxes)

        if mAP == True: 
            return image, bboxes
        
        image, bboxes = preprocess_image(np.copy(image), [self.input_size, self.input_size], np.copy(bboxes))
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        label = [
            np.zeros((self.output_sizes[i], self.output_sizes[i], self.anchor_per_scale, 5 + self.num_classes))
            for i in range(3)
        ]

        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
