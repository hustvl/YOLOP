# YOLOP by hustvl, MIT License
"""
PyTorch Hub models https://pytorch.org/hub/
Usage:
    import torch
    model = torch.hub.load(xxx, xxx)
"""

import torch
import torchvision.transforms as transforms

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


if __name__ == '__main__':
    model = _create(device='cpu')
    import numpy as np
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    imgs = np.zeros((640, 640, 3),dtype=np.float32)
    img = transform(imgs).to("cpu")
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    img = torch.tensor(img ,dtype=torch.float32)

    results = model(img)  # batched inference
    