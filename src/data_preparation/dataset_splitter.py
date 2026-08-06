import os
import random
import shutil
import yaml

random.seed(42)
source_folder=[
    "data/interim/thermal_yolo_print1/obj_train_data",
    "data/interim/thermal_yolo_print2/obj_train_data",
    "data/interim/thermal_yolo_print3/obj_train_data"
]

# collect all images
all_images=[]
for folder in source_folder:
    images=[
         os.path.join(folder,f)
         for f in os.listdir(folder)
         if f.endswith(".png")
    ]
    all_images.extend(images)
print(f"Total images found: {len(all_images)}")
random.shuffle(all_images)

with open("configs/config.yaml","r") as file:
    config=yaml.safe_load(file)

train_ratio=config["split"]["train_ratio"]
val_ratio=config["split"]["val_ratio"]
test_ratio=config["split"]["test_ratio"]

total=len(all_images)
train_end=int(total*train_ratio)
val_end=train_end+int(total*val_ratio)

train_images=all_images[:train_end]
val_images=all_images[train_end:val_end]
test_images=all_images[val_end:]

print("Train:" ,len(train_images))
print("Val:" ,len(val_images))
print("Test:" ,len(test_images)) 

#create a yolo output folder
splits={
    "train":train_images,
    "val":val_images,
    "test":test_images

}


for split_name in splits.keys():
    os.makedirs(f"data/processed/images/{split_name}", exist_ok=True)
    os.makedirs(f"data/processed/labels/{split_name}", exist_ok=True)

# Copy images and labels
for split_name, image_list in splits.items():

    for image_path in image_list:

        image_name = os.path.basename(image_path)

        # Get dataset name to avoid duplicate filenames
        dataset_name = os.path.basename(
         os.path.dirname(
        os.path.dirname(image_path)
         )
            )
        
        new_image_name = f"{dataset_name}_{image_name}"
        new_label_name = new_image_name.replace(".png", ".txt")

        label_path = image_path.replace(".png", ".txt")

        shutil.copy(
            image_path,
            f"data/processed/images/{split_name}/{new_image_name}"
        )

        shutil.copy(
            label_path,
            f"data/processed/labels/{split_name}/{new_label_name}"
        )

print("Dataset split completed.")