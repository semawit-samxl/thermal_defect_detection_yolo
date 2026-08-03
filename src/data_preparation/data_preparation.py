import os
import zipfile
import yaml

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

raw_data_path = config["data"]["raw_data_path"]
interim_data_path = config["data"]["interim_data_path"]

os.makedirs(interim_data_path, exist_ok=True)

zip_files = [f for f in os.listdir(raw_data_path) if f.endswith(".zip")]

for zip_file in zip_files:
    zip_path = os.path.join(raw_data_path, zip_file)

    dataset_name = os.path.splitext(zip_file)[0]
    extract_path = os.path.join(interim_data_path, dataset_name)

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"{zip_file}: Extracted to {extract_path}")

print("Extraction completed.")