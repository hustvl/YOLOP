# YOLOP by hustvl, MIT License
"""
PyTorch Hub models https://pytorch.org/hub/
Usage:
    import torch
    model = torch.hub.load(xxx, xxx)
"""

import torch
import cv2
import torchvision.transforms as transforms
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box

def _create(pretrained=True, weights = "./weights/End-to-end.pth", device=None):
    """Creates YOLOP model
    Arguments:
        pretrained (bool): load pretrained weights into the model
        wieghts (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters
    Returns:
        YOLOP pytorch model
    """
    
    
    from lib.utils.utils import select_device
    from lib.config import cfg
    from lib.models import get_net

    device = select_device(device = device)
    model = get_net(cfg)
    if pretrained:
        checkpoint = torch.load(weights, map_location= device)
        model.load_state_dict(checkpoint['state_dict'])
    return model.to(device)


def makeborder(img, stride=32, pad=114):
    shape = img.shape[:2]
    print(shape[0], shape[1])
    dw, dh = np.mod(shape[1], stride), np.mod(shape[0], stride)
    dw, dh = dw/2, dh/2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad)  # add border
    return img



if __name__ == '__main__':
    model = _create(device='cpu')
    model.eval()
    import numpy as np
    from numpy import random

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    filename = "test.jpg"
    img = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img_det=img.copy()
    img = makeborder(img, stride=32)
    img = transform(img)
    img = img.unsqueeze(0) if img.ndimension() == 3 else img
    det_out, da_seg_out,ll_seg_out= model(img)
