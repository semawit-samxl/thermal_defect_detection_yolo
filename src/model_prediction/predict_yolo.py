from ultralytics import YOLO


def predict():

    model = YOLO("runs/detect/train/weights/best.pt")

    results = model.predict(
        source="data/processed/images/test",
        save=True
    )

    return results


if __name__ == "__main__":
    predict()