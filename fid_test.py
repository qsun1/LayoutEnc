import os

from torchmetrics.image.fid import FrechetInceptionDistance as FID
import torchvision
import torch
import torch.utils.data
from tqdm import tqdm

#from taming.data.faceshq import FFHQTrain, FFHQValidation
from taming.data.base import ImagePaths

class CustomDataset(ImagePaths):
    def __init__(self, root):
        paths = os.listdir(root)
        paths = [os.path.join(root, fname) for fname in paths]
        super().__init__(paths, size=256)

def convert_to_uint8(images_float):
    return (torch.clamp(images_float * 0.5 + 0.5, 0., 1.) * 255.).to(dtype=torch.uint8)

#dataset_gen = CustomDataset(root='/data/sunq/layout2img/taming-transformers-master/inference_single/052515_100_1.0')
dataset_gen = CustomDataset(root='/data/sunq/layout2img/taming-transformers-master/inference_many/060684_100_1.0')
#dataset_trn = FFHQTrain(256)
dataset_trn = CustomDataset(root='/data2/sunq/inst-seg/data/coco/val2017')


gen_loader = torch.utils.data.DataLoader(dataset_gen, batch_size=128, num_workers=16)
trn_loader = torch.utils.data.DataLoader(dataset_trn, batch_size=128, num_workers=16)

fid_module = FID(feature=2048).to('cuda') 

for batch in tqdm(trn_loader):
    imgs = batch['image'].permute(0, 3, 1, 2)
    imgs = convert_to_uint8(imgs).cuda()
    fid_module.update(imgs, real=True)

for batch in tqdm(gen_loader):
    imgs = batch['image'].permute(0, 3, 1, 2)
    imgs = convert_to_uint8(imgs).cuda()
    fid_module.update(imgs, real=False)

fid = fid_module.compute().item()
print(f'FID: {fid:.6f}')