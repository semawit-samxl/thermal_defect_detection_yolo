import yaml
import mlflow
from ultralytics import YOLO




def train_model():

   with open("configs/params.yaml", "r") as file:
    params = yaml.safe_load(file)

    mlflow.set_tracking_uri("sqlite:///mlfow.db")
    mlflow.set_experiment("thermal_data_defect_detection_yolo")

    #start mlflow tracking
    with mlflow.start_run():


     #load traiining parametrs 
     mlflow.log_params(params["training"])

     #load model
     model = YOLO(params["training"]["model"])

     # train model
      
     results= model.train(
        data="configs/data.yaml",
        epochs=params["training"]["epochs"],
        imgsz=params["training"]["image_size"],
        batch=params["training"]["batch_size"],
        patience=params["training"]["patience"],
        device=params["training"]["device"]

      ) 
      
    return results


if __name__ == "__main__":
    train_model()