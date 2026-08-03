import yaml
from ultralytics import YOLO


def train_model():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    model = YOLO(params["training"]["model"])

    model.train(
        data="configs/data.yaml",
        epochs=params["training"]["epochs"],
        imgsz=params["training"]["image_size"],
        batch=params["training"]["batch_size"],
        patience=params["training"]["patience"],
        device=params["training"]["device"]
    )


if __name__ == "__main__":
    train_model()