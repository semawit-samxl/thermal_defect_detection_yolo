import os
import yaml
import torch

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import draw_bounding_boxes

from src.model_training.faster_rcnn_model import (create_model)


def predict():

    # Load configuration
    with open(
        "configs/rcnn_params.yaml",
        "r"
    ) as file:

        params = yaml.safe_load(file)

    # Select device
    device = torch.device(
        params["training"]["device"]
    )

    # Create model
    model = create_model()

    # Load checkpoint
    checkpoint = torch.load(
        params["paths"]["best_model"],
        map_location=device
    )

    # Load weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Move model to device
    model.to(device)

    # Set evaluation mode
    model.eval()

    # Define paths
    test_dir = (
        f"{params['data']['processed_data_path']}"
        "/images/test"
    )

    output_dir = (
        "predictions/faster_rcnn"
    )

    # Create output directory
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Disable gradients
    with torch.no_grad():

        # Process images
        for image_name in os.listdir(
            test_dir
        ):

            if not image_name.endswith(
                ".png"
            ):
                continue

            # Build image path
            image_path = os.path.join(
                test_dir,
                image_name
            )

            # Load image
            image = Image.open(
                image_path
            ).convert("RGB")

            # Convert to tensor
            image_tensor = ToTensor()(
                image
            ).to(device)

            # Run inference
            prediction = model(
                [image_tensor]
            )[0]

            # Filter predictions
            mask = (
                prediction["scores"]
                >=
                params["prediction"][
                    "confidence_threshold"
                ]
            )

            boxes = prediction[
                "boxes"
            ][mask]

            labels = prediction[
                "labels"
            ][mask]

            scores = prediction[
                "scores"
            ][mask]

            # Create label text
            label_names = []

            for label, score in zip(
                labels,
                scores
            ):

                class_name = (
                    params["classes"][
                        label.item() - 1
                    ]
                )

                label_names.append(
                    f"{class_name} "
                    f"{score:.2f}"
                )

            # Draw bounding boxes
            result_image = draw_bounding_boxes(
                (
                    image_tensor * 255
                ).byte().cpu(),
                boxes.cpu(),
                labels=label_names,
                colors="red",
                width=2
            )

            # Convert tensor to numpy image
            result_image = (
                result_image
                .permute(1, 2, 0)
                .numpy()
            )

            # Save prediction image
            Image.fromarray(
                result_image
            ).save(
                os.path.join(
                    output_dir,
                    image_name
                )
            )

            print(
                f"Processed: "
                f"{image_name}"
            )

            print(
                f"Predictions saved to "
                f"{output_dir}"
            )

            return output_dir


if __name__ == "__main__":

    predict()