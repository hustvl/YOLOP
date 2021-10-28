# YOLOP by hustvl, MIT License
dependencies = ['torch']
import torch
from lib.utils.utils import select_device
from lib.config import cfg
from lib.models import get_net
from pathlib import Path
import os

def yolop(pretrained=True, device="cpu"):
    """Creates YOLOP model
    Arguments:
        pretrained (bool): load pretrained weights into the model
        wieghts (int): the url of pretrained weights
        device (str): cuda device i.e. 0 or 0,1,2,3 or cpu
    Returns:
        YOLOP pytorch model
    """
    device = select_device(device = device)
    model = get_net(cfg)
    if pretrained:
        path = os.path.join(Path(__file__).resolve().parent, "weights/End-to-end.pth")
        checkpoint = torch.load(path, map_location= device)
        model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    return model




