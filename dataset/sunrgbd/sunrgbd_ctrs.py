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
import torch
import torch.utils.data as data
import torch.nn as nn
import itertools


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


class SUNRGBD_CTRS:

    def __init__(self, data_dir, split, depth_transform, label_type, margin, n_neg, is_embedding_set, dataset_threshold) -> None:
        self.sample_stack = {}
        self.margin = margin
        self.n_neg = n_neg
        self.split = split
        self.is_embedding_set = is_embedding_set

        for modal in ['rgb', 'depth']:
            self.sample_stack[modal] = glob(join(data_dir, split, modal, '*.jpg' if modal == 'rgb' else '*.png'))
            assert len(self.sample_stack[modal]) > 0, f"No {modal} images found in {data_dir}/{split}/{modal}"
            self.sample_stack[modal].sort()

        for i in range(len(self.sample_stack['rgb'])):
            assert splitext(basename(self.sample_stack['rgb'][i]))[0] == splitext(basename(self.sample_stack['depth'][i]))[0], f"RGB and depth images not matched: {self.sample_stack['rgb'][i]}, {self.sample_stack['depth'][i]}"

        assert split in ['train'], "Only support train split, use SUNRGBD for building test split"
        if split in ['train']:
            self.shared_transform = A.Compose([A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=cv2.INTER_CUBIC), A.HorizontalFlip(p=0.5)], additional_targets={'image_depth': 'image'})
            self.rgb_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            if depth_transform == '01':
                self.depth_transform = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
            elif depth_transform == 'rgb':
                self.depth_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.label_type = label_type
        if self.label_type == 'clip_vitb32':
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

            self.positives = [] # 5000
            for i, label in enumerate(self.clip_labels):
                positive = np.where(self.clip_labels == label)[0]                                      # find same-label samples
                positive = np.delete(positive, np.where(positive == i)[0])                             # delete self
                self.positives.append(positive)

            self.negatives = []
            for i, label in enumerate(self.clip_labels):
                negative = np.where(self.clip_labels != label)[0]                                      # find different-label samples
                self.negatives.append(negative)

        if self.is_embedding_set == False:
            self.embedding_cache = None  # torch.rand((self.__len__(), 512)) # ([5000, 512]) # how much memory does it take? 5000 * 512 * 4 = 10MB

    def _load_sample(self, index):
        input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_depth = cv2.imread(self.sample_stack['depth'][index], cv2.IMREAD_GRAYSCALE)

        transformed_image = self.shared_transform(image=input_rgb, image_depth=input_depth)
        image_rgb = self.rgb_transform(image=transformed_image['image'])
        image_depth = self.depth_transform(image=np.repeat(transformed_image['image_depth'][:, :, np.newaxis], repeats=3, axis=-1))
        image_depth_array = image_depth['image']
        image_rgb_array = image_rgb['image']
        image_rgb_array = np.transpose(image_rgb_array, (2, 0, 1))
        image_depth_array = np.transpose(image_depth_array, (2, 0, 1))

        return torch.from_numpy(image_rgb_array), torch.from_numpy(image_depth_array)


    def __getitem__(self, index):
        if self.split in ['train']:
            if self.is_embedding_set == True:
                rgb, depth = self._load_sample(index)
                # class_name = basename(self.sample_stack['rgb'][index]).split('__')[0]
                # class_id = class_name_to_id[class_name]
                return rgb, depth, index

            else:
                p_indices = self.positives[index]
                if len(p_indices) < 1:
                    return None
                a_emb = self.embedding_cache[index]
                p_emb = self.embedding_cache[p_indices]
                dist_ap = torch.norm(a_emb - p_emb, dim=1, p=None) # Np NOTE
                d_p, inds_p = dist_ap.topk(1, largest=False) # e.g., d_p:tensor([8.4966, 8.7970, 8.8521]), NOTE introcude some randomness here?
                index_p = p_indices[inds_p].item()

                n_indices = self.negatives[index]
                # n_indices = np.random.choice(self.negatives[index], self.n_negative_subset)                # randomly choose potential_negatives
                n_emb = self.embedding_cache[n_indices]                                                              # (Np, D)
                dist_an = torch.norm(a_emb - n_emb, dim=1, p=None)                                            # Np
                d_n, inds_n = dist_an.topk(self.n_neg * 100 if len(n_indices) > self.n_neg * 100 else len(n_indices), largest=False) # e.g., d_p:tensor([8.4966, 8.7970, 8.8521]),
                loss_positive_indices = d_n < d_p + self.margin                                                # [True, True, ...] tensor
                if torch.sum(loss_positive_indices) < 1:
                    return None
                n_indices_valid = inds_n[loss_positive_indices][:self.n_neg].numpy()                                # tensor -> numpy: a[tensor(5)] = 1, a[numpy(5)]=array([1])
                index_n = n_indices[n_indices_valid]

                a_rgb, a_depth = self._load_sample(index) # (C, H, W), (C, H, W)
                p_rgb, p_depth = self._load_sample(index_p) # (C, H, W), (C, H, W)
                n_rgb = torch.stack([self._load_sample(ind)[0] for ind in index_n])  # (n_neg, C, H, W])
                n_depth = torch.stack([self._load_sample(ind)[1] for ind in index_n], 0) # (n_neg, C, H, W])

                return a_rgb, p_rgb, n_rgb, a_depth, p_depth, n_depth, [self.clip_labels[index], self.clip_labels[index_p]] + [self.clip_labels[x] for x in index_n]


    def __len__(self):
        return len(self.sample_stack['rgb'])

    def collate_fn(self, batch_input):
        batch = list(filter(lambda x: x is not None, batch_input))
        if len(batch) == 0:
            return None, None, None

        a_rgb, p_rgb, n_rgb, a_depth, p_depth, n_depth, labels = zip(*batch)
        num_negs = data.dataloader.default_collate([x.shape[0] for x in n_rgb]) # ([B, C, H, W]) = ([C, H, W])

        a_rgb = data.dataloader.default_collate(a_rgb)     # ([B, C, H, W]) = ([C, H, W]) + ...
        p_rgb = data.dataloader.default_collate(p_rgb)     # ([B, C, H, W]) = ([C, H, W]) + ...
        n_rgb = torch.cat(n_rgb, 0)                   # ([B, C, H, W]) = ([C, H, W]) + ...
        a_depth = data.dataloader.default_collate(a_depth) # ([B, C, H, W]) = ([C, H, W]) + ...
        p_depth = data.dataloader.default_collate(p_depth) # ([B, C, H, W]) = ([C, H, W]) + ...
        n_depth = torch.cat(n_depth, 0)               # ([B, C, H, W]) = ([C, H, W]) + ...

        labels = list(itertools.chain(*labels))

        return torch.cat((a_rgb, p_rgb, n_rgb, a_depth, p_depth, n_depth)), num_negs, labels

if __name__ == '__main__':
    dataset_train = SUNRGBD_CTRS(split='train', data_dir='sunrgbd/RGB_JPG', depth_transform='rgb')
    dataset_test = SUNRGBD_CTRS(split='test', data_dir='sunrgbd/RGB_JPG', depth_transform='rgb')
    print(f"Train: {len(dataset_train)},  Test: {len(dataset_test)}")
    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4, collate_fn=dataset_train.collate_fn)
    criterion = nn.TripletMarginLoss(margin=0.1, p=2, reduction='sum')
    loss_rgb_triplet = 0
    loss_depth_triplet = 0
    for imgs, nums_neg, labels in loader_train:
        # imgs: (44, 3, 224, 224), nums_neg: ([2, 2, 5, 5]), labels: ([22])
        if imgs is None:
            continue
        B = 4
        nums_neg_total = torch.sum(nums_neg)                                                              # 20
        a_rgb_emb, p_rgb_emb, n_rgb_emb, a_depth_emb, p_depth_emb, n_depth_emb = torch.split(imgs, [B, B, nums_neg_total, B, B, nums_neg_total])
        for i, num_neg in enumerate(nums_neg):
            for n in range(num_neg):
                negIx = (torch.sum(nums_neg[:i]) + n).item()
                loss_rgb_triplet += criterion(a_rgb_emb[i:i + 1], p_rgb_emb[i:i + 1], n_rgb_emb[negIx:negIx + 1])
                loss_depth_triplet += criterion(a_depth_emb[i:i + 1], p_depth_emb[i:i + 1], n_depth_emb[negIx:negIx + 1])

        loss_rgb_triplet /= nums_neg_total.float() # normalise by actual number of negatives
        loss_depth_triplet /= nums_neg_total.float() # normalise by actual number of negatives
        pass                 # ([32, 3, 224, 224]), ([32, 224, 224]), ([32])
        break
