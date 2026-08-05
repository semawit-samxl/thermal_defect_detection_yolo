import yaml
import torch

from torch.utils.data import (DataLoader)

from torchmetrics.detection.mean_ap import (MeanAveragePrecision)

from src.data_preparation.faster_rcnn_dataset import (ThermalDataset,collate_fn)

from src.model_training.faster_rcnn_model import (create_model)


def evaluate():

    # Load configuration
    with open(
        "configs/rcnn_params.yaml",
        "r"
    ) as file:

        params = yaml.safe_load(file)

    # Select device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Create validation dataset
    dataset = ThermalDataset(
        image_dir=
        f"{params['data']['processed_data_path']}/images/val",

        label_dir=
        f"{params['data']['processed_data_path']}/labels/val"
    )

    # Create validation dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=params["evaluation"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    model = create_model()

    # Load best checkpoint
    checkpoint = torch.load(
        "models/best_faster_rcnn.pth",
        map_location=device
    )

 