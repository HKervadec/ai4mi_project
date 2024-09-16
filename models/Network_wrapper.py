from models.ENet import ENet
# from models.Other_Net import other_net
from models.ShallowNet import shallowCNN

def get_model(model_name):

    # implementation based on:
    # https://github.com/riadhassan/UDBRNet/blob/main/Network/Network_wrapper.py

    architecture = {
        "shallowCNN": shallowCNN,
        "ENet": ENet,
        # "Other_Net": other_net
    }
    
    model = architecture[model_name]

    return model