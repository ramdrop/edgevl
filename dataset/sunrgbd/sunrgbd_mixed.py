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


class_id_to_name = {
    0: "bathroom",
    1: "bedroom",
    2: "classroom",
    3: "computer_room",
    4: "conference_room",
    5: "corridor",
    6: "dining_area",
    7: "dining_room",
    8: "discussion_area",
    9: "furniture_store",
    10: "home_office",
    11: "kitchen",
    12: "lab",
    13: "lecture_theatre",
    14: "library",
    15: "living_room",
    16: "office",
    17: "rest_space",
    18: "study_space"
}
class_name_to_id = {v: k for k, v in class_id_to_name.items()}

class_names = set(class_id_to_name.values())

# shared_transform = A.Compose([A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=cv2.INTER_CUBIC), A.HorizontalFlip(p=0.5)], additional_targets={'image_depth': 'image'})
# rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
# depth_transform = A.Compose([A.NoOp()])

# test_shared_transform = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_CUBIC), A.CenterCrop(224, 224)], additional_targets={'image_depth': 'image'})
# test_rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
# test_depth_transform = A.Compose([A.NoOp()])

class SUNRGBD_MIXED:
    def __init__(self, data_dir='sunrgbd/organized-set/RGB_JPG', split='train', depth_transform='rgb', label_type='gt', mixed=False) -> None:
        self.sample_stack = {}
        self.mixed = mixed
        
        for modal in ['rgb', 'depth', 'mixed']:
            self.sample_stack[modal] = glob(join(data_dir, split, modal, '*.jpg' if modal == 'rgb' else '*.png'))
            assert len(self.sample_stack[modal]) > 0, f"No {modal} images found in {data_dir}/{split}/{modal}"
            self.sample_stack[modal].sort()

        for i in range(len(self.sample_stack['rgb'])):
            assert splitext(basename(self.sample_stack['rgb'][i]))[0] == splitext(basename(self.sample_stack['depth'][i]))[0], f"RGB and depth images not matched: {self.sample_stack['rgb'][i]}, {self.sample_stack['depth'][i]}"

        if split in ['train', 'sub_train']:
            self.shared_transform = A.Compose([A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=cv2.INTER_CUBIC), A.HorizontalFlip(p=0.5)], additional_targets={'image_depth': 'image'})
            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        elif split in ['sub_val', 'test']:
            self.shared_transform = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_CUBIC), A.CenterCrop(224, 224)], additional_targets={'image_depth': 'image'})
            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.label_type = label_type
        if self.label_type == 'clip_vitb32':
            self.clip_predictions_df = pd.read_csv(join(data_dir, 'clip_vitb32_prediction.csv'), index_col=0)

    def _load_sample(self, index):
        if self.mixed == True:
            input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['mixed'][index]), cv2.COLOR_BGR2RGB)
        else:
            input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_depth = cv2.imread(self.sample_stack['depth'][index], cv2.IMREAD_GRAYSCALE)

        return input_rgb, input_depth

    def __getitem__(self, index):
        input_rgb, input_depth = self._load_sample(index)

        transformed_image = self.shared_transform(image=input_rgb, image_depth=input_depth)
        image_rgb = self.rgb_transform(image=transformed_image['image'])
        image_depth = self.depth_transform(image=np.repeat(transformed_image['image_depth'][:, :, np.newaxis], repeats=3, axis=-1))
        # image_depth_array = image_depth['image'].astype(np.float32)/255.0
        image_depth_array = image_depth['image']
        image_rgb_array = image_rgb['image']
        image_rgb_array = np.transpose(image_rgb_array, (2, 0, 1))
        image_depth_array = np.transpose(image_depth_array, (2, 0, 1))

        if self.label_type == 'gt':
            # ground truth labels
            class_name = basename(self.sample_stack['rgb'][index]).split('__')[0]
            class_id = class_name_to_id[class_name]
            return image_rgb_array, image_depth_array, class_id
        elif self.label_type == 'clip_vitb32':
            # clip pseudo labels
            class_id = self.clip_predictions_df.loc[basename(self.sample_stack['rgb'][index]), 'class_id']
            similarity = self.clip_predictions_df.loc[basename(self.sample_stack['rgb'][index]), 'similarity']
            return image_rgb_array, image_depth_array, [class_id, similarity]

    def __len__(self):
        return len(self.sample_stack['rgb'])


if __name__ == '__main__':
    dataset_train = SUNRGBD(split='sub_train', data_dir='sunrgbd/organized-set/RGB_JPG', depth_transform='rgb')
    dataset_val = SUNRGBD(split='sub_val', data_dir='sunrgbd/organized-set/RGB_JPG', depth_transform='rgb')
    dataset_test = SUNRGBD(split='test', data_dir='sunrgbd/organized-set/RGB_JPG', depth_transform='rgb')
    print(f"Train: {len(dataset_train)}, Val: {len(dataset_val)}, Test: {len(dataset_test)}")
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
    for image_rgb_array, image_depth_array, class_id in loader_train:
        pass                                     # ([32, 3, 224, 224]), ([32, 224, 224]), ([32])
        break
