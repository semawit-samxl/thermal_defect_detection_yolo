import os
import sys

sys.path.append(os.getcwd())

from src.model_evaluation.evaluate_fast_rcnn import (
    evaluate
)


if __name__ == "__main__":

    evaluate()