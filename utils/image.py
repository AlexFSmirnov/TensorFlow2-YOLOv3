import cv2
import random
import numpy as np

def preprocess_image(image, target_size, bboxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw / w, ih / h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh: nh + dh, dw: nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if bboxes is None:
        return image_paded
    else:
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dh
        return image_paded, bboxes

def random_augmentations(image, bboxes, prob=0.5):
    for aug in [horizontal_flip, crop, translate]:
        if random.random() <= prob:
            image, bboxes = aug(np.copy(image), np.copy(bboxes))
    return image, bboxes

def horizontal_flip(image, bboxes):
    _, w, _ = image.shape
    image = image[:, ::-1, :]
    bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

    return image, bboxes

def crop(image, bboxes):
    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w - max_bbox[2]
    max_d_trans = h - max_bbox[3]

    crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
    crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
    crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
    crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

    image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes

def translate(image, bboxes):
    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w - max_bbox[2]
    max_d_trans = h - max_bbox[3]

    tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
    ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

    M = np.array([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h))

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes