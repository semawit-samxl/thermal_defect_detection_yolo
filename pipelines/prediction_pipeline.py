import sys
import os

sys.path.append(os.getcwd())

from src.model_prediction.predict import predict

if __name__ == "__main__":
    predict()