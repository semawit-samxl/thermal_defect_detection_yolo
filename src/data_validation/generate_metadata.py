import os
import json
import yaml


def generate_metadata():

    # Load configuration
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    interim_path = config["data"]["interim_data_path"]

    metadata = {
        "dataset": {
            "name": "thermal_defect_detection",
            "image_type": "thermal",
            "annotation_format": "YOLO",
            "total_images": 0,
            "total_labels": 0,
            "total_annotated_frames": 0,
            "class_distribution": {},
            "print_statistics": {}
        }
    }

    class_counts = {}

    datasets = [
        folder
        for folder in os.listdir(interim_path)
        if os.path.isdir(os.path.join(interim_path, folder))
    ]

    print("Datasets found:", datasets)

    for dataset in datasets:

        data_path = os.path.join(
            interim_path,
            dataset,
            "obj_train_data"
        )

        if not os.path.exists(data_path):
            continue

        images = [
            f for f in os.listdir(data_path)
            if f.endswith(".png")
        ]

        labels = [
            f for f in os.listdir(data_path)
            if f.endswith(".txt")
        ]

        annotated_frames = 0

        for label_file in labels:

            label_path = os.path.join(data_path, label_file)

            if os.path.getsize(label_path) > 0:

                annotated_frames += 1

                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line in lines:

                    if not line.strip():
                        continue

                    class_id = int(line.split()[0])

                    class_counts[class_id] = (
                        class_counts.get(class_id, 0) + 1
                    )

        metadata["dataset"]["print_statistics"][dataset] = {
            "images": len(images),
            "labels": len(labels),
            "annotated_frames": annotated_frames
        }

        metadata["dataset"]["total_images"] += len(images)
        metadata["dataset"]["total_labels"] += len(labels)
        metadata["dataset"]["total_annotated_frames"] += annotated_frames

    class_names = config.get("classes", [])

    for class_id, count in class_counts.items():

        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        metadata["dataset"]["class_distribution"][class_name] = count

    os.makedirs(
        "artifacts/data_validation",
        exist_ok=True
    )

    output_path = (
        "artifacts/data_validation/dataset_metadata.json"
    )

    # This is the dump code
    with open(output_path, "w") as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print(f"Metadata saved to: {output_path}")


if __name__ == "__main__":
    generate_metadata()