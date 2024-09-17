from models.ENet import ENet
from models.UDBRNet import UDBRNet
from models.ShallowNet import shallowCNN

def get_model(model_name):

    # implementation based on:
    # https://github.com/riadhassan/UDBRNet/blob/main/Network/Network_wrapper.py

    architecture = {
        "shallowCNN": shallowCNN,
        "ENet": ENet,
        "UDBRNet": UDBRNet
    }
    
    model = architecture[model_name]

    return model