import os
import yaml
import torch

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import draw_bounding_boxes

from src.model_training.faster_rcnn_model import (
    create_model
)


def predict():

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

    # Create model
    model = create_model()

    # Load checkpoint
    checkpoint = torch.load(
        params["paths"]["best_model"],
        map_location=device,
        weights_only=False
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Move model to device
    model.to(device)

    # Evaluation mode
    model.eval()

    # Test dataset directory
    test_dir = (
        f"{params['data']['processed_data_path']}"
        "/images/test"
    )

    # Output directory
    output_dir = (
        "artifacts/predictions/faster_rcnn"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Disable gradients
    with torch.no_grad():

        for image_name in os.listdir(
            test_dir
        ):

            if not image_name.endswith(
                ".png"
            ):
                continue

            # Load image
            image_path = os.path.join(
                test_dir,
                image_name
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            image_tensor = ToTensor()(
                image
            ).to(device)

            # Run inference
            prediction = model(
                [image_tensor]
            )[0]

            # Confidence filtering
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
                    f"({score.item():.2f})"
                )

            # Draw boxes and confidence scores
            result_image = draw_bounding_boxes(
                (
                    image_tensor * 255
                ).byte().cpu(),
                boxes.cpu(),
                labels=label_names,
                colors="red",
                width=2
            )

            # Convert tensor back to image
            result_image = (
                result_image
                .permute(1, 2, 0)
                .numpy()
            )

            # Save prediction image
            output_path = os.path.join(
                output_dir,
                image_name
            )

            Image.fromarray(
                result_image
            ).save(
                output_path
            )

            print(
                f"Processed: {image_name}"
            )

    print(
        f"\nPredictions saved to:"
    )

    print(output_dir)

    return output_dir


def main():

    predict()


if __name__ == "__main__":

    main()