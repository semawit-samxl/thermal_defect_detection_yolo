# validate.py

import yaml

def validate(map50):

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    threshold = params["validation"]["min_map50"]

    return map50 >= threshold