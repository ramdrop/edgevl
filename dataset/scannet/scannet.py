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

def remove_spaces_around_slash(input_string):
    if input_string == 'computercluster':
        return 'computer cluster'
    parts = input_string.split('/')
    cleaned_parts = [part.strip() for part in parts]
    result_string = '/'.join(cleaned_parts)
    return result_string

class SCANNET:

    class_id_to_name = {
        0: "apartment",
        1: "bathroom",
        2: "bedroom/hotel",
        3: "bookstore/library",
        4: "classroom",
        5: "closet",
        6: "computer cluster",
        7: "conference room",
        8: "copy/mail room",
        9: "dining room",
        10: "game room",
        11: "gym",
        12: "hallway",
        13: "kitchen",
        14: "laundry room",
        15: "living room/lounge",
        16: "lobby",
        17: "office",
        18: "stairs",
        19: "storage/basement/garage",
        20: "misc"
        }

    class_name_to_id = {v: k for k, v in class_id_to_name.items()}
    clip_descriptions = [f"a photo of {determine_prefix(convert_string(x))}" for x in class_id_to_name.values()]

    def __init__(self, data_dir='', split='train', depth_transform='rgb', label_type='gt', is_subset=False, dataset_threshold=0.0) -> None:
        print(f"==> Loading {split} data from {data_dir}, Depth transform: {depth_transform}, Label type: {label_type}")
        self.sample_stack = {}
        if split in ['train']:
            self.shared_transform = A.Compose([A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=cv2.INTER_CUBIC), A.HorizontalFlip(p=0.5)], additional_targets={'image_depth': 'image'})
            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            split_meta = join(data_dir, 'train_every_100th.txt')
            with open(split_meta) as f:
                file_list = f.readlines()
            self.sample_stack['rgb']  = [join(data_dir, 'train/rgb', x.strip()+'.jpg') for x in file_list]
            self.sample_stack['depth'] = [join(data_dir, 'train/depth', x.strip()+'.png') for x in file_list]
            label_files = [join(data_dir, 'train/scene_class', x.strip()[:-6]+'.txt') for x in file_list]
            self.sample_stack['labels'] = []
            for label_file in label_files:
                with open(label_file) as f:
                    self.sample_stack['labels'].append(self.class_name_to_id[remove_spaces_around_slash(f.readline())])
            assert len(self.sample_stack['rgb']) > 0, f"No rgb images found in {data_dir}/{split}/rgb"
            assert len(self.sample_stack['depth']) > 0, f"No depth images found in {data_dir}/{split}/depth"

            if is_subset == True:
                self.clip_predictions_df = pd.read_csv(join(data_dir, 'pseudo_labels.csv'), index_col=0)
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

        elif split in ['valid', 'test']:
            split = 'valid' if split == 'test' else 'valid'
            self.shared_transform = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_CUBIC)], additional_targets={'image_depth': 'image'})
            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            split_meta = join(data_dir, f'{split}_every_100th.txt')
            with open(split_meta) as f:
                file_list = f.readlines()
            self.sample_stack['rgb']  = [join(data_dir, f'{split}/rgb', x.strip()+'.jpg') for x in file_list]
            self.sample_stack['depth'] = [join(data_dir, f'{split}/depth', x.strip()+'.png') for x in file_list]
            label_files = [join(data_dir, f'{split}/scene_class', x.strip()[:-6]+'.txt') for x in file_list]
            self.sample_stack['labels'] = []
            for label_file in label_files:
                with open(label_file) as f:
                    self.sample_stack['labels'].append(self.class_name_to_id[remove_spaces_around_slash(f.readline())])

            assert len(self.sample_stack['rgb']) > 0, f"No rgb images found in {data_dir}/{split}/rgb"
            assert len(self.sample_stack['depth']) > 0, f"No depth images found in {data_dir}/{split}/depth"

        for i in range(len(self.sample_stack['rgb'])):
            assert splitext(basename(self.sample_stack['rgb'][i]))[0] == splitext(basename(self.sample_stack['depth'][i]))[0], f"RGB and depth images not matched: {self.sample_stack['rgb'][i]}, {self.sample_stack['depth'][i]}"

        self.label_type = label_type
        if self.label_type == 'pseudo':
            assert exists(join(data_dir, 'pseudo_labels.csv')), f"Pseudo labels not found in {data_dir}"
            self.clip_predictions_df = pd.read_csv(join(data_dir, 'pseudo_labels.csv'), index_col=0)        

    def _load_sample(self, index):
        input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_rgb = cv2.resize(input_rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
        input_depth = cv2.imread(self.sample_stack['depth'][index], -1)
        # Normalize depth to [0, 255]
        input_depth = ((input_depth - np.min(input_depth)) / (np.max(input_depth) - np.min(input_depth)) * 255.0).astype(np.uint8)

        return input_rgb, input_depth

    def __getitem__(self, index):
        input_rgb, input_depth = self._load_sample(index)

        transformed_image = self.shared_transform(image=input_rgb, image_depth=input_depth)
        image_rgb = self.rgb_transform(image=transformed_image['image'])
        image_depth = self.depth_transform(image=np.repeat(transformed_image['image_depth'][:, :, np.newaxis], repeats=3, axis=-1))
        image_depth_array = image_depth['image']
        image_rgb_array = image_rgb['image']
        image_rgb_array = np.transpose(image_rgb_array, (2, 0, 1))
        image_depth_array = np.transpose(image_depth_array, (2, 0, 1))

        if self.label_type == 'gt':
            return image_rgb_array, image_depth_array, self.sample_stack['labels'][index]
        elif self.label_type == 'pseudo':
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
    dataset_train = SCANNET(split='train', data_dir='../../dbs/scannet', depth_transform='rgb', label_type='gt')
    dataset_val = SCANNET(split='valid', data_dir='../../dbs/scannet', depth_transform='rgb', label_type='gt')
    print(f"Train: {len(dataset_train)}, Val: {len(dataset_val)}")
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
    for image_rgb_array, image_depth_array, class_id in loader_train:
        pass                                     # ([32, 3, 224, 224]), ([32, 224, 224]), ([32])
        break
