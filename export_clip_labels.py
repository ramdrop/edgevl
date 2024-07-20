#%%
import albumentations as A
import cv2
from glob import glob
from os.path import basename, dirname, exists, join, splitext
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dataset.eurosat.eurosat import EUROSAT
from dataset.nyu2.nyu2 import NYU2
from dataset.sunrgbd.sunrgbd import SUNRGBD
from dataset.scannet.scannet import SCANNET
import open_clip
from utils.misc import setup_seed, get_grad_norm, schedule_device
import re
import os
import json
import shutil
import argparse
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
from dataset.build_dataset import get_dataset
import sys


# sys.argv = ['']              # !! NOTE Comment this out when running in the terminal !!
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='')
args = parser.parse_args()


# Load the model
setup_seed(0)
torch.cuda.set_device(schedule_device())
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_modal_name = ("ViT-g-14", "laion2b_s34b_b88k")
model, _, preprocess = open_clip.create_model_and_transforms(clip_modal_name[0], pretrained=clip_modal_name[1], device=device, cache_dir='_cache')

#%%

DATASET = args.dataset               # 'eurosat', 'nyu2', 'sunrgbd', 'scannet'
dataset = get_dataset(DATASET)(split='train', data_dir=join('dbs', DATASET),  label_type='file')
dataset_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

print(f'dataset: {DATASET}')
print('num of samples:', len(dataset))

if  DATASET in ['sunrgbd', 'scannet', 'nyu2']:
    categories = 'preprocess_dataset/indoor_categories.txt'
    prefix = 'a photo of'
elif DATASET == 'eurosat':
    categories = 'preprocess_dataset/satellite_categories.txt'
    prefix = 'a satellite image of'

# read general labels from scene.txt:
with open(categories, 'r') as f:
    general_categories = f.readlines()
general_categories = [x.strip() for x in general_categories]
general_categories = list(filter(None, general_categories))
# remove empty strings
print(general_categories)


def convert_string(string):
    # Split camel case words using regex
    words = re.findall('[A-Z][a-z]*', string)

    # Convert words to lowercase and join with a space
    return ' '.join(words).lower()


def determine_prefix(string):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if string[0] in vowels:
        return "an " + string
    else:
        return "a " + string


general_categories_descriptions = [f"{prefix} {determine_prefix(convert_string(x))}" for x in general_categories]

print('num of indoor_categories:', len(general_categories_descriptions))

tokenizer = open_clip.get_tokenizer(clip_modal_name[0])
text_inputs = torch.cat([tokenizer(x) for x in general_categories_descriptions]).to(device) # ([N, 77])
with torch.no_grad():
    text_features = model.encode_text(text_inputs)                                                       # ([19, 512])
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#%%
file_to_pred = {}
for batch_idx, (rgb_imgs, files) in enumerate(tqdm(dataset_loader, leave=False)):
    # get a sample from dataloader
    with torch.no_grad():
        image_features = model.encode_image(rgb_imgs.to(device)) # ([1, 512])

    image_features /= image_features.norm(dim=-1, keepdim=True) # ([32, 1024])
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) # ([32, 51]) <= ([32, 1024]) * # ([1024, 51])
    for i in range(len(similarity)):
        values, indices = similarity[i].topk(5)
        file_to_pred[files[i]] = [indices[0].item(), values[0].item()]


#%% export predictions
output_dir = 'preprocess_dataset/pseudo_labels'
if not exists(join(output_dir, DATASET)):
    os.makedirs(join(output_dir, DATASET))

df = pd.DataFrame.from_dict(file_to_pred, orient='index', columns=['class_id', 'similarity'])
df.to_csv(join(output_dir, DATASET, 'pseudo_labels.csv'), index=True)
