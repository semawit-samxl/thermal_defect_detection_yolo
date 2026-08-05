from src.model_training.train_yolo import train_model
from src.model_prediction.predict_yolo import predict
from src.model_evaluation.evaluate_yolo import evaluate
from src.model_validation.validate import validate
from src.model_analysis.analyze import analyze


def main():

    print("Starting Model Training...")
    train_model()

    print("Generating Predictions...")
    predict()

    print("Evaluating Model...")
    metrics = evaluate()

    map50 = metrics.box.map50

    print(f"mAP50: {map50}")

    if validate(map50):
        print("Model Accepted")

    else:
        print("Model Rejected")
        analyze()


if __name__ == "__main__":
    main()