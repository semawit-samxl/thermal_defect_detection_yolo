import os

datasets=[
    "thermal_yolo_print1",
    "thermal_yolo_print2",
    "thermal_yolo_print3"
]

base_path="data/interim"

for dataset in datasets:
    data_path=os.path.join(base_path,dataset,"obj_train_data")
    images=[f for f in os.listdir(data_path) if f.endswith(".png")]
    labels=[f for f in os.listdir(data_path) if f.endswith(".txt")]

    annotated=0
    for label in labels:
        label_path=os.path.join(data_path,label)

        if os.path.getsize(label_path)>0:
            annotated+=1
    print(f"\n{dataset}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Annotated Frames:{annotated}")
