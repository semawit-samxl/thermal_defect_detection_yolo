import sys
import os

sys.path.append(os.getcwd())

from src.model_prediction.predict_yolo import predict

if __name__ == "__main__":
    predict()