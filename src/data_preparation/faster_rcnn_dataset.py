import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class ThermalDataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir
    ):

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = sorted(
            [
                file
                for file in os.listdir(image_dir)
                if file.endswith(".png")
            ]
        )

    def __len__(self):

        return len(self.images)

    def __getitem__(
        self,
        idx
    ):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        label_path = os.path.join(
            self.label_dir,
            image_name.replace(
                ".png",
                ".txt"
            )
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):

            with open(
                label_path,
                "r"
            ) as file:

                lines = file.readlines()

            for line in lines:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])

                box_width = float(values[3])
                box_height = float(values[4])

                xmin = (
                    x_center - box_width / 2
                ) * width

                ymin = (
                    y_center - box_height / 2
                ) * height

                xmax = (
                    x_center + box_width / 2
                ) * width

                ymax = (
                    y_center + box_height / 2
                ) * height

                boxes.append(
                    [
                        xmin,
                        ymin,
                        xmax,
                        ymax
                    ]
                )

                # Background = 0
                # Class 1 -> label 1
                # Class 2 -> label 2
                labels.append(
                    class_id + 1
                )

        if len(boxes) == 0:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        else:

            boxes = torch.as_tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.as_tensor(
                labels,
                dtype=torch.int64
            )

        area = (
            (boxes[:, 2] - boxes[:, 0]) *
            (boxes[:, 3] - boxes[:, 1])
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros(
                (len(boxes),),
                dtype=torch.int64
            )
        }

        image = ToTensor()(image)

        return image, target


def collate_fn(batch):

    return tuple(zip(*batch))


def create_dataset():

    dataset = ThermalDataset(

        image_dir="data/processed/images/train",

        label_dir="data/processed/labels/train"
    )

    return dataset


def create_dataloader():

    dataset = create_dataset()

    dataloader = DataLoader(

        dataset,

        batch_size=4,

        shuffle=True,

        collate_fn=collate_fn
    )

    return dataloader


def main():

    dataloader = create_dataloader()

    images, targets = next(
        iter(dataloader)
    )

    print(
        f"Batch Size: {len(images)}"
    )

    print(
        f"Image Shape: {images[0].shape}"
    )

    print(
        f"Target Keys: {targets[0].keys()}"
    )

    print(
        f"Boxes Shape: {targets[0]['boxes'].shape}"
    )

    print(
        f"Labels Shape: {targets[0]['labels'].shape}"
    )

    print(
        f"Area Shape: {targets[0]['area'].shape}"
    )

    print(
        f"Iscrowd Shape: {targets[0]['iscrowd'].shape}"
    )


if __name__ == "__main__":

    main()