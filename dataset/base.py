import albumentations as A
import cv2
import pandas as pd
from os.path import join, basename, dirname
import numpy as np


class Base:

    def __init__(self, data_dir='sunrgbd', split='train', depth_transform='rgb', label_type='gt', is_subset=False, dataset_threshold=0.0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.split = split
        self.label_type = label_type
        self.depth_transform = depth_transform
        self.is_subset = is_subset
        self.dataset_threshold = dataset_threshold

        self._parse_file_list()
        self._parse_class_id()
        self._init_transforms()
        self._refine_dataset()

        return


    def _refine_dataset(self):
        if self.split in ['train'] and self.is_subset == True:
            self.clip_predictions_df = pd.read_csv(join(self.data_dir, f'{self.label_type}.csv'), index_col=0)
            self.class_id_stack, self.similarity_stack = np.array(self.class_id_stack), np.array(self.similarity_stack)
            # confident_mask = self.similarity_stack > self.dataset_threshold
            confident_mask = self.similarity_stack > np.percentile(self.similarity_stack, 100 * (1 - self.dataset_threshold))
            print(f"Confident samples ratio: {np.mean(confident_mask):.3f}")
            self.class_id_stack, self.similarity_stack = self.class_id_stack[confident_mask], self.similarity_stack[confident_mask]
            self.sample_stack['rgb'], self.sample_stack['depth'] = np.array(self.sample_stack['rgb'])[confident_mask], np.array(self.sample_stack['depth'])[confident_mask]


    def _init_transforms(self):
        depth_transform_lut = {
            '01': A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))]),
            'rgb': A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
        }

        if self.split in ['train']:
            if self.label_type == 'file':
                # in this mode, we are exporting clip labels, so we use the test transform
                self.shared_T = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_CUBIC)], additional_targets={'image_depth': 'image'})
            else:
                self.shared_T = A.Compose([
                    A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=cv2.INTER_CUBIC),
                    A.HorizontalFlip(p=0.5),
                ], additional_targets={'image_depth': 'image'})
        elif self.split in ['test', 'valid']:
            self.shared_T = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_CUBIC)], additional_targets={'image_depth': 'image'})
        else:
            print(f"Unknown split: {self.split}")
            raise NotImplementedError

        self.depth_T = depth_transform_lut[self.depth_transform]
        self.rgb_T = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


    def _parse_file_list(self):
        raise NotImplementedError

    def _parse_gt_id(self):
        raise NotImplementedError

    def _parse_clip_id(self):

        print(f"==> Loading pseudo labels from {self.data_dir}")
        self.clip_predictions_df = pd.read_csv(join(self.data_dir, f'{self.label_type}.csv'), index_col=0)
        self.class_id_stack, self.similarity_stack = [], []

        for index in range(len(self.sample_stack['rgb'])):
            class_id = self.clip_predictions_df.loc[self.sample_stack['rgb'][index], 'class_id']
            similarity = self.clip_predictions_df.loc[self.sample_stack['rgb'][index], 'similarity']
            self.class_id_stack.append(class_id)
            self.similarity_stack.append(similarity)

        return

    def _parse_class_id(self):
        # print(self.split, self.label_type)
        if self.label_type in ['gt']:
            self._parse_gt_id()
        elif self.label_type in ['clip_vitb32', 'pseudo_labels']:
            self._parse_clip_id()

        return


    def _load_sample(self, index):

        input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_depth = cv2.imread(self.sample_stack['depth'][index], cv2.IMREAD_GRAYSCALE)

        return input_rgb, input_depth


    def _apply_transform(self, input_rgb, input_depth):

        transformed_image = self.shared_T(image=input_rgb, image_depth=input_depth)
        image_rgb = self.rgb_T(image=transformed_image['image'])
        image_depth = self.depth_T(image=np.repeat(transformed_image['image_depth'][:, :, np.newaxis], repeats=3, axis=-1))
        image_depth_array = image_depth['image']
        image_rgb_array = image_rgb['image']
        image_rgb_array = np.transpose(image_rgb_array, (2, 0, 1))
        image_depth_array = np.transpose(image_depth_array, (2, 0, 1))

        return image_rgb_array, image_depth_array


    def __getitem__(self, index):
        rgb_np, depth_np = self._load_sample(index)
        rgb_np, depth_np = self._apply_transform(rgb_np, depth_np)

        if self.label_type in ['gt']:
            return rgb_np, depth_np, self.class_id_stack[index]
        elif self.label_type in ['clip_vitb32', 'pseudo_labels']:
            return rgb_np, depth_np, [self.class_id_stack[index], self.similarity_stack[index]]
        elif self.label_type in ['file']:
            return rgb_np, self.sample_stack['rgb'][index]
        else:
            raise NotImplementedError


    def __len__(self):
        return len(self.sample_stack['rgb'])

    def __str__(self):
        return f"Base class"
