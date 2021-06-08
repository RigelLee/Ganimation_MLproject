import torch
import numpy as np
from PIL import Image


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_np = image_tensor[0].cpu().float().numpy()
    im = np2im(image_np, imtype).resize((80, 80), Image.ANTIALIAS)
    return np.array(im)

def np2im(image_np, imtype=np.uint8):
    if image_np.shape[0] == 1:
        image_np = np.tile(image_np, (3, 1, 1))
        
    image_np = (np.transpose(image_np, (1, 2, 0)) / 2. + 0.5) * 255.0
    
    image_np = image_np.astype(imtype)
    im = Image.fromarray(image_np)
    
    return im