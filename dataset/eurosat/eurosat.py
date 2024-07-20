#%%
from glob import glob
from os.path import basename, dirname, exists, join, splitext

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
import cv2
import pandas as pd
import re


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

class EUROSAT:

    def __init__(self, data_dir='eurosat', split='train', depth_transform='rgb', label_type='gt', is_subset=False, dataset_threshold=0.0) -> None:

        self.class_id_to_name = {0:'AnnualCrop', 1:'Forest', 2:'HerbaceousVegetation', 3:'Highway', 4:'Industrial', 5:'Pasture', 6:'PermanentCrop', 7:'Residential', 8:'River', 9:'SeaLake',}
        self.class_name_to_id = {v: k for k, v in self.class_id_to_name.items()}
        self.clip_descriptions = [f"a satellite image of {determine_prefix(convert_string(x))}" for x in self.class_id_to_name.values()]
        # clip_descriptions = [f"a satellite image of a {x}" for x in class_id_to_name.values()]

        self.sample_stack = {}

        # read from txt file
        files = []
        with open(join(data_dir, split+'.txt'), 'r') as f:
            for line in f:
                files.append(line.strip())
        files.sort()

        self.sample_stack = {}
        for modal in ['rgb', 'depth']:
            self.sample_stack[modal] = [join(data_dir, 'EuroSAT_RGB' if modal == 'rgb' else 'EuroSAT_SWIR', file) for file in files]
        for i in range(len(self.sample_stack['rgb'])):
            assert basename(self.sample_stack['rgb'][i]) == basename(self.sample_stack['depth'][i]), 'rgb and depth file name should be the same'

        if split in ['train']:
            if is_subset == True:
                self.clip_predictions_df = pd.read_csv(join(data_dir, 'clip_vitb32_prediction.csv'), index_col=0)
                self.clip_labels = []
                self.similarities = []
                for file in self.sample_stack['rgb']:
                    self.clip_labels.append(int(self.clip_predictions_df.loc[file]['class_id']))
                    self.similarities.append(self.clip_predictions_df.loc[file]['similarity'])
                self.clip_labels = np.array(self.clip_labels)
                self.similarities = np.array(self.similarities)

                confident_mask = self.similarities > dataset_threshold
                print(f"Confident samples ratio: {np.mean(confident_mask):.3f}")
                self.clip_labels = self.clip_labels[confident_mask]
                self.similarities = self.similarities[confident_mask]
                self.sample_stack['rgb'] = np.array(self.sample_stack['rgb'])[confident_mask]
                self.sample_stack['depth'] = np.array(self.sample_stack['depth'])[confident_mask]

            self.shared_transform = A.Compose([A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=cv2.INTER_CUBIC), A.HorizontalFlip(p=0.5)], additional_targets={'image_depth': 'image'})
            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        elif split in ['test']:
            # self.shared_transform = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_CUBIC), A.CenterCrop(224, 224)], additional_targets={'image_depth': 'image'})
            self.shared_transform = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_CUBIC)], additional_targets={'image_depth': 'image'})

            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.label_type = label_type
        if self.label_type == 'clip_vitb32':
            self.clip_predictions_df = pd.read_csv(join(data_dir, 'clip_vitb32_prediction.csv'), index_col=0)
        # elif self.label_type == 'gt':
        #     self.labels = []
        #     for idx in idxs:
        #         with open(join(data_dir, split, 'scene_class', f'{idx:04d}.txt'), 'r') as f:
        #             self.labels.append(self.class_name_to_id[f.read().strip()])

    def _load_sample(self, index):
        input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_depth = cv2.imread(self.sample_stack['depth'][index], cv2.IMREAD_UNCHANGED)

        return input_rgb, input_depth

    def __getitem__(self, index):
        input_rgb, input_depth = self._load_sample(index)

        transformed_image = self.shared_transform(image=input_rgb, image_depth=input_depth)
        image_rgb = self.rgb_transform(image=transformed_image['image'])

        input_depth = np.repeat(transformed_image['image_depth'][:, :, np.newaxis], repeats=3, axis=-1)
        # input_depth = input_depth.astype(np.float32) / input_depth.max() * 255
        image_depth = self.depth_transform(image=input_depth)

        image_depth_array = image_depth['image']
        image_rgb_array = image_rgb['image']
        image_rgb_array = np.transpose(image_rgb_array, (2, 0, 1))
        image_depth_array = np.transpose(image_depth_array, (2, 0, 1))

        if self.label_type == 'gt':
            # eurosat/EuroSAT_RGB/AnnualCrop/AnnualCrop_1.jpg
            class_name = self.sample_stack['rgb'][index].split('/')[-2]
            return image_rgb_array, image_depth_array, self.class_name_to_id[class_name]
        elif self.label_type == 'clip_vitb32':
            # clip pseudo labels
            class_id = self.clip_predictions_df.loc[self.sample_stack['rgb'][index], 'class_id']
            similarity = self.clip_predictions_df.loc[self.sample_stack['rgb'][index], 'similarity']
            return image_rgb_array, image_depth_array, [class_id, similarity]
        elif self.label_type == 'file':
            return image_rgb_array, self.sample_stack['rgb'][index]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.sample_stack['rgb'])


if __name__ == '__main__':
    dataset_train = EUROSAT(split='train', data_dir='../../dbs/eurosat', depth_transform='rgb')
    dataset_test = EUROSAT(split='test', data_dir='../../dbs/eurosat', depth_transform='rgb')
    print(f"Train: {len(dataset_train)}, Test: {len(dataset_test)}")
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
    for image_rgb_array, image_depth_array, class_id in loader_train:
        pass                                     # ([32, 3, 224, 224]), ([32, 224, 224]), ([32])
        break
