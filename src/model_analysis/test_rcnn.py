import sys
import os

sys.path.append(os.getcwd())

from src.model_training.faster_rcnn_model import create_model

model = create_model()

print(model)