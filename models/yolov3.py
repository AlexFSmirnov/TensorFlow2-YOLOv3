import cv2
import numpy as np
import tensorflow as tf
from utils.model_creation import create_keras_model
from utils.detection import image_preprocess, postprocess_boxes, nms, draw_bbox
from common.enums import ModelPurpose, WeightType

class YOLOv3:
    def __init__(self, num_classes, purpose=ModelPurpose.Detection, input_size=416, channels=3, tiny=False):
        self.num_classes = num_classes
        self.purpose = purpose
        self.input_size = input_size
        self.channels = channels
        self.tiny = tiny

        self.model = create_keras_model(
            num_classes,
            input_size,
            channels,
            tiny,
            is_for_training=(purpose == ModelPurpose.Training),
        )

    def load_weights(self, path, type):
        if type == WeightType.Darknet:
            self.load_darknet_weights(path)
        elif type == WeightType.Checkpoint:
            self.model.load_weights(path)
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
                
                conv_layer = self.model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = self.model.get_layer(bn_layer_name)
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

    def train(self):
        if self.purpose != ModelPurpose.Training:
            raise Exception(f'Model was not created with purpose "{self.purpose.name}". Create it with purpose "{ModelPurpose.Training.name}" instead.')

        pass

    def detect_image(self, image_path, classes, output_path=None, score_threshold=0.8, iou_threshold=0.45, show=False):
        if self.purpose != ModelPurpose.Detection:
            raise Exception(f'Model was not created with purpose "{self.purpose.name}". Create it with purpose "{ModelPurpose.Detection.name}" instead.')

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = image_preprocess(np.copy(original_image), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, self.input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        image = draw_bbox(original_image, bboxes, classes=classes)

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
