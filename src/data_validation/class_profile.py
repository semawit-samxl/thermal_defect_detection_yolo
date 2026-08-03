import os

datasets = [
    "thermal_yolo_print1",
    "thermal_yolo_print2",
    "thermal_yolo_print3"
]

base_path = "data/interim"

class_counts = {
    0: 0,  # Blob
    1: 0,  # Underextrusion
    2: 0   # Sagging
}

for dataset in datasets:
    data_path = os.path.join(base_path, dataset, "obj_train_data")

    labels = [f for f in os.listdir(data_path) if f.endswith(".txt")]

    for label_file in labels:
        label_path = os.path.join(data_path, label_file)

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.strip():
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

print("\nClass Distribution")
print(f"Blob: {class_counts[0]}")
print(f"Underextrusion: {class_counts[1]}")
print(f"Sagging: {class_counts[2]}")