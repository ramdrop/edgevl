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
    # Replace underscores with spaces
    converted_string = string.replace("_", " ")
    return converted_string

def determine_prefix(string):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if string[0] in vowels:
        return "an " + string
    else:
        return "a " + string

class NYU2:

    class_id_to_name = {
        0: 'basement',
        1: 'bathroom',
        2: 'bedroom',
        3: 'bookstore',
        4: 'cafe',
        5: 'classroom',
        6: 'computer_lab',
        7: 'conference_room',
        8: 'dinette',
        9: 'dining_room',
        10: 'excercise_room',
        11: 'foyer',
        12: 'furniture_store',
        13: 'home_office',
        14: 'home_storage',
        15: 'indoor_balcony',
        16: 'kitchen',
        17: 'laundry_room',
        18: 'living_room',
        19: 'office',
        20: 'office_kitchen',
        21: 'playroom',
        22: 'printer_room',
        23: 'reception_room',
        24: 'student_lounge',
        25: 'study',
        26: 'study_room'
        }

    class_name_to_id = {v: k for k, v in class_id_to_name.items()}
    clip_descriptions = [f"a photo of {determine_prefix(convert_string(x))}" for x in class_id_to_name.values()]

    def __init__(self, data_dir='nyu2', split='train', depth_transform='rgb', label_type='gt') -> None:
        self.sample_stack = {}

        # read from txt file
        idxs = []
        with open(join(data_dir, split+'.txt'), 'r') as f:
            for line in f:
                idxs.append(int(line.strip()))
        self.sample_stack = {}
        for modal in ['rgb', 'depth']:
            self.sample_stack[modal] = [join(data_dir, split, modal, f'{idx:04d}.png') for idx in idxs]

        for i in range(len(self.sample_stack['rgb'])):
            assert splitext(basename(self.sample_stack['rgb'][i]))[0] == splitext(basename(self.sample_stack['depth'][i]))[0], f"RGB and depth images not matched: {self.sample_stack['rgb'][i]}, {self.sample_stack['depth'][i]}"

        if split in ['train']:
            if label_type == 'file':
                # in this mode, we are exporting clip labels, so we use the test transform
                self.shared_transform = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_CUBIC)], additional_targets={'image_depth': 'image'})
            else:
                self.shared_transform = A.Compose([A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=cv2.INTER_CUBIC), A.HorizontalFlip(p=0.5)], additional_targets={'image_depth': 'image'})
            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        elif split in ['test']:
            self.shared_transform = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_CUBIC)], additional_targets={'image_depth': 'image'})
            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.label_type = label_type
        if self.label_type == 'clip_vitb32':
            self.clip_predictions_df = pd.read_csv(join(data_dir, 'clip_vitb32_prediction.csv'), index_col=0)
        elif self.label_type == 'gt':
            self.labels = []
            for idx in idxs:
                with open(join(data_dir, split, 'scene_class', f'{idx:04d}.txt'), 'r') as f:
                    self.labels.append(self.class_name_to_id[f.read().strip()])


    def _load_sample(self, index):
        input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_depth = cv2.imread(self.sample_stack['depth'][index], cv2.IMREAD_UNCHANGED)

        return input_rgb, input_depth


    def __getitem__(self, index):
        input_rgb, input_depth = self._load_sample(index)

        transformed_image = self.shared_transform(image=input_rgb, image_depth=input_depth)
        image_rgb = self.rgb_transform(image=transformed_image['image'])

        input_depth = np.repeat(transformed_image['image_depth'][:, :, np.newaxis], repeats=3, axis=-1)
        input_depth = input_depth.astype(np.float32) / input_depth.max() * 255
        image_depth = self.depth_transform(image=input_depth)

        image_depth_array = image_depth['image']
        image_rgb_array = image_rgb['image']
        image_rgb_array = np.transpose(image_rgb_array, (2, 0, 1))
        image_depth_array = np.transpose(image_depth_array, (2, 0, 1))

        if self.label_type == 'gt':
            return image_rgb_array, image_depth_array, self.labels[index]
        elif self.label_type == 'clip_vitb32':
            # clip pseudo labels
            class_id = self.clip_predictions_df.loc[basename(self.sample_stack['rgb'][index]), 'class_id']
            similarity = self.clip_predictions_df.loc[basename(self.sample_stack['rgb'][index]), 'similarity']
            return image_rgb_array, image_depth_array, [class_id, similarity]
        elif self.label_type == 'file':
            return image_rgb_array, self.sample_stack['rgb'][index]
        else:
            raise ValueError(f"Unknown label type: {self.label_type}")

    def __len__(self):
        return len(self.sample_stack['rgb'])


if __name__ == '__main__':
    dataset_train = NYU2(split='train', data_dir='../../dbs/nyu2', depth_transform='rgb')
    dataset_train.class_id_to_name
    # dataset_test = NYU2(split='test', data_dir='nyu2', depth_transform='rgb')
    # print(f"Train: {len(dataset_train)}, Test: {len(dataset_test)}")
    # loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
    # for image_rgb_array, image_depth_array, class_id in loader_train:
    #     pass                                     # ([32, 3, 224, 224]), ([32, 224, 224]), ([32])
    #     break
