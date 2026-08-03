import os
import zipfile 
import yaml

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

raw_data_path = config.get("raw_data_path", "data/raw")

files=os.listdir(raw_data_path)
zip_files=[file for file in files if file.endswith(".zip")]

print(f" Found {len(zip_files)} Zip Files.")

for file in zip_files:
    zip_path=os.path.join(raw_data_path,file)

    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"{file}: Valid Zip file")

            image_files=[ 
                name for name in zip_ref.namelist()
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
            print(f"{file}: contains {len(image_files)} image files.")
    except:
        print(f"{file}: Invalid or Corrupted file")

