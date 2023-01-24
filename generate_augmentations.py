import os
import random
import numpy as np
import torch
from training.augment import AugmentPipe
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('path_to_imgs', metavar='DIR', nargs='?', default='datasets/dogs/',
                    help='path to dataset')
parser.add_argument('destination_path', metavar='DIR', nargs='?', default='datasets/augmented_dogs/',
                    help='path to save augmented dataset')
parser.add_argument("--per_image_augmentations", default=400, type=int, help="Number of augmentations per image")
parser.add_argument("--augmentation_prob", default=0.15, type=float, help="Probability of applying an augmentation")


if __name__ == '__main__':
    args = parser.parse_args()
    path_to_imgs = args.path_to_imgs
    destination_path = args.destination_path
    per_image_augmentations = args.per_image_augmentations

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
        j = 0
    else:
        j = len(os.listdir(os.path.join(destination_path, os.listdir(destination_path)[0]))) - 1

    print(f"Generating augmentations for {path_to_imgs} and saving them to {destination_path}")
    t = transforms.ToPILImage()

    dataset = datasets.ImageFolder(path_to_imgs, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


    for img, l in tqdm(loader):
        for i in tqdm(range(per_image_augmentations)):

            aug_pipe = AugmentPipe(p=args.augmentation_prob, 
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
