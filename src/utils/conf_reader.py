import yaml
import argparse
import torch

def get_config():
    try:
        conf_file = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    except FileNotFoundError:
        try:
            conf_file = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
        except FileNotFoundError:
            conf_file = yaml.load(open('src/config.yaml'), Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", default=0.005, help="The learning rate")
    parser.add_argument("--load_model", default=None, help="path to the model to load")
    args = parser.parse_args()

    args_dict = {
        'load_model': str(args.load_model),
        'lr': float(args.learning_rate),
        'device': device,
    }

    return {**conf_file, **args_dict}
