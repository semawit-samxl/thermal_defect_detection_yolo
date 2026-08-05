import yaml

import sys
import os
sys.path.append(os.getcwd())
with open("configs/rcnn_params.yaml","r") as file:
    params=yaml.safe_load(file)


    print(params["training"]["batch_size"])
    print(params["classes"])



