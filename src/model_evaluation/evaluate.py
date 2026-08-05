# evaluate.py

import mlflow

from ultralytics import YOLO


def evaluate():

    model = YOLO("runs/detect/train/weights/best.pt")

    metrics = model.val(
        data="configs/data.yaml"
      )
    mlflow.set_tracking_uri("sqlite:///mlfow.db")
    
    with mlflow.start_run():
     
      mlflow.log_metric("map50",metrics.box.map50)
      mlflow.log_metric("map50_90",metrics.box.map)
      mlflow.log_metric("precision",metrics.box.mp)
      mlflow.log_metric("recall",metrics.box.mr)

    return metrics