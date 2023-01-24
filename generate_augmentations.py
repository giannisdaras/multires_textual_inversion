import os
import random
import numpy as np
import torch
from training.augment import AugmentPipe
from torchvision import datasets, transforms


path_to_imgs = ""
destination_path = ""

t = transforms.ToPILImage()

dataset = datasets.ImageFolder(path_to_imgs, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

classes_keys = list(dog_class_dict.keys())

j = 0
for img, l in loader:
    for i in range(2):

        aug_pipe = AugmentPipe(p=0.15, 
                    xflip=1, 
                    yflip=1,  
                    translate_int=1, 
                    scale=1, 
                    rotate_frac=1, 
                    aniso=1, 
                    translate_frac=1, 
                #    brightness=1, 
                #    contrast=1, 
                #    lumaflip=1, 
                    hue=1, 
                    saturation=1
        )

        concept_dir = destination_path + list(dataset.class_to_idx.keys())[l]
        concept_path = os.path.join(concept_dir)
    
        if not os.path.exists(concept_path):
            os.mkdir(concept_path)

        img = img.clip(min=-1, max=1)
        aug_img_t = aug_pipe(img)
        aug_img = t(aug_img_t[0][0])
        img_path = f"{concept_path}/{j}.jpg"
        aug_img.save(img_path)
        j +=1