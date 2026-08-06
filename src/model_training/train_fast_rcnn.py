import os
import yaml
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



from torch.utils.data import DataLoader

from src.data_preparation.faster_rcnn_dataset import (
    ThermalDataset,
    collate_fn
)

from src.model_training.faster_rcnn_model import (
    create_model
)


def train_model():

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

    # Create dataset
    dataset = ThermalDataset(
        image_dir="data/processed/images/train",
        label_dir="data/processed/labels/train"
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=params["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=params["training"].get(
            "num_workers",
            0
        )
    )

    # Create model
    model = create_model()

    # Move model to device
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["training"]["learning_rate"]
    )

    # Create model directory
    os.makedirs(
        "models",
        exist_ok=True
    )

    # Initialize best loss
    best_loss = float("inf")

    # Print dataset information
    print(
        f"Dataset Size: {len(dataset)}"
    )

    print(
        f"Batches: {len(dataloader)}"
    )

    print(
        f"Device: {device}"
    )

    # Start training
    for epoch in range(
        params["training"]["epochs"]
    ):

        # Set training mode
        model.train()

        # Initialize epoch loss
        epoch_loss = 0.0

        # Iterate through batches
        for batch_idx, (
            images,
            targets
        ) in enumerate(dataloader):

            # Move images to device
            images = [
                image.to(device)
                for image in images
            ]

            # Move targets to device
            targets = [
                {
                    k: v.to(device)
                    for k, v in target.items()
                }
                for target in targets
            ]

            # Forward pass
            loss_dict = model(
                images,
                targets
            )

            # Calculate total loss
            total_loss = sum(
                loss
                for loss in loss_dict.values()
            )

            # Clear gradients
            optimizer.zero_grad()

            # Compute gradients
            total_loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            epoch_loss += total_loss.item()

            # Print batch progress
            if (
                batch_idx % 50 == 0
            ):

                print(
                    f"Epoch "
                    f"{epoch + 1} "
                    f"Batch "
                    f"{batch_idx}/"
                    f"{len(dataloader)} "
                    f"Loss: "
                    f"{total_loss.item():.4f}"
                )

        # Compute average loss
        avg_loss = (
            epoch_loss /
            len(dataloader)
        )

        # Print epoch summary
        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{params['training']['epochs']} "
            f"Average Loss: "
            f"{avg_loss:.4f}"
        )

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "loss": avg_loss
            },
            "models/last_faster_rcnn.pth"
        )

        # Save best checkpoint
        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                {
                    "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                            "loss": avg_loss
                },
                "models/best_faster_rcnn.pth"
            )

        print( 
              f"New best model saved "
              f"with loss "
              f"{best_loss:.4f}"

        )
        # Print completion message
        print(
            "Training completed"
        )

        print(
            f"Best loss: "
            f"{best_loss:.4f}"
        )   