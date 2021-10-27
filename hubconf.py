# YOLOP by hustvl, MIT License
dependencies = ['torch']
import torch
from lib.utils.utils import select_device
from lib.config import cfg
from lib.models import get_net

def yolop(pretrained=True, weights = "https://github.com/hustvl/YOLOP/blob/main/weights/End-to-end.pth", device="cpu"):
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
        model.load_state_dict(torch.hub.load_state_dict_from_url(weights, map_location=device, progress=False))
    return model




