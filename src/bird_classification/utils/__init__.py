from pathlib import Path
import yaml

def get_config_and_param():
    config_filepath = Path("src/bird_classification/utils/config.yaml")
    params_filepath = Path("src/bird_classification/utils/params.yaml")
    # print(config_filepath)

    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)
        # print(config)
    with open(params_filepath, 'r') as file:
        params = yaml.safe_load(file)
        # print(params)
    return config,params

CONFIG_FILE, PARAMS_FILE = get_config_and_param()
# print(PARAMS_FILE)