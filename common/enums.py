from enum import Enum

ModelPurpose = Enum('ModelPurpose', 'training detection transfer')

WeightType = Enum('WeightType', 'darknet checkpoint')

DatasetType = Enum('DatasetType', 'train test')

CheckpointSaveMode = Enum('CheckpointSaveMode', 'all best last')
