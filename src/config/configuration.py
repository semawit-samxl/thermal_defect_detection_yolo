import yaml
with open("configs/config.yaml","r") as file:
    config=yaml.safe_load(file)
    print(config)