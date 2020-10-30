from models.yolov3 import YOLOv3
from utils.general import get_classes_from_dataset
from common.enums import ModelPurpose, WeightType

classes = get_classes_from_dataset('dataset')

model = YOLOv3(len(classes), purpose=ModelPurpose.Detection)
model.load_weights('pre_trained_weights/custom/checkpoint/yolov3_custom', type=WeightType.Checkpoint)
model.detect_image('dataset/test/img-2.jpg', classes, show=True)
