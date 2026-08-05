import sys
import os

sys.path.append(os.getcwd())

from src.model_evaluation.evaluate import evaluate

if __name__ == "__main__":
    evaluate()