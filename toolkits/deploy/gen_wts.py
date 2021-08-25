import torch
import struct

# Initialize
device = torch.device('cpu')
# Load model
model = torch.load('yolov5s.pt', map_location=device)  # ['model'].float()
# load to FP32
model.to(device).eval()

f = open('yolov5s.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')

f.close()
