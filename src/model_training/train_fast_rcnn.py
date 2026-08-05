import torch

from torchvision.models.detection import (fasterrcnn_resnet50_fpn)

from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor)

#get preptrained model
def get_model():

    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    num_classes = 3

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor =FastRCNNPredictor(in_features,num_classes)
    

    return model


def main():

    model = get_model()

    print(model)

    print(
        "\nFaster R-CNN loaded successfully."
    )


if __name__ == "__main__":

    main()