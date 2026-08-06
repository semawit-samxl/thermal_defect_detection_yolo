import torchvision

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

#here we replaced the classifer head network based on our classes =2 (blob and underextrusion and background num class= 3)

def create_model():

    num_classes = 3

    model = (
        torchvision.models.detection
        .fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
    )

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            num_classes
        )
    )

    return model
