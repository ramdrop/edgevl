#%%
import os
from os.path import basename, dirname, exists, join, splitext
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import open_clip
# open_clip.list_pretrained()
from dataset.build_dataset import get_dataset
import argparse
from utils.misc import setup_seed, get_grad_norm, schedule_device
import shlex
import sys

launch_command = ' '.join(shlex.quote(arg) for arg in sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', choices=['eurosat', 'nyu2', 'sunrgbd', 'scannet'], help='dataset to evaluate')
parser.add_argument('--modal', type=str, default='', choices=['depth', 'rgb'], help='set modal to evaluate')
parser.add_argument('--clip_version', type=str, default='', choices=['G', 'B'])
args = parser.parse_args()


export_dir = f'visulization/feature_inspect_{args.dataset}/clip'
if not exists(export_dir):
    os.makedirs(export_dir)
    
# Load the model
torch.cuda.set_device(schedule_device())
device = "cuda" if torch.cuda.is_available() else "cpu"
if args.clip_version == 'G':
    clip_modal_name = ("ViT-g-14", "laion2b_s34b_b88k")
elif args.clip_version == 'B':    
    clip_modal_name = ('ViT-B-16', 'laion2b_s34b_b88k')
else:
    raise ValueError('clip_version should be either G or B')
model, _, preprocess = open_clip.create_model_and_transforms(clip_modal_name[0], pretrained=clip_modal_name[1], device=device, cache_dir='_cache')

#%% genereate predictions
DATASET = args.dataset       # 'eurosat', 'nyu2', 'sunrgbd', 'scannet'
MODAL = args.modal           # 'rgb', 'depth'
batch_size = 32
dataset = get_dataset(DATASET)(split='test', data_dir=join('dbs', DATASET), depth_transform='rgb', label_type='gt')
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
print(f"dataset: {DATASET}, labels length: {len(dataset.clip_descriptions)}, modal: {MODAL}")

tokenizer = open_clip.get_tokenizer(clip_modal_name[0])

text_inputs = torch.cat([tokenizer(x) for x in dataset.clip_descriptions]).to(device) # ([N, 77])
with torch.no_grad():
    text_features = model.encode_text(text_inputs)                                    # ([19, 512])
text_features /= text_features.norm(dim=-1, keepdim=True)


cnt_correct = np.zeros(3)
for batch_idx, (image_rgb_array, image_depth_array, class_id) in enumerate(tqdm(dataset_loader)):
    if MODAL == 'rgb':
        image_input = image_rgb_array.to(device)   # ([1, 3, 224, 224])
    elif MODAL == 'depth':
        image_input = image_depth_array.to(device) # ([1, 3, 224, 224])

    with torch.no_grad():
        image_features = model.encode_image(image_input) # ([1, 512])

    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) # ([1, 19])
                                                                            # p1_3_5 = np.array([1, int(0.3*len(dataset.clip_descriptions)), int(0.5*len(dataset.clip_descriptions))])
    p1_3_5 = np.array([1, 3, 5])
    for i in range(len(similarity)):
        values, indices = similarity[i].topk(max(p1_3_5))
        prediction = indices.cpu().numpy()
        for j in range(len(p1_3_5)):
            if class_id[i].item() in prediction[:p1_3_5[j]]:
                cnt_correct[j:] += 1
                break


acc = cnt_correct / len(dataset)
print(f"Acc@1: @1:{acc[0]:.4f}, @3:{acc[1]:.4f}, @5:{acc[2]:.4f}")
with open(join(export_dir, 'acc.txt'), 'a') as f:
    f.write(f"{launch_command}\n")
    f.write(f"Acc@1: @1:{acc[0]:.4f}, @3:{acc[1]:.4f}, @5:{acc[2]:.4f}\n\n")
