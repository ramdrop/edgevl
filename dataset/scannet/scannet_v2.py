#%%
from glob import glob
from os.path import basename, dirname, exists, join, splitext
import numpy as np
import cv2

from dataset.base import Base


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


class SCANNET(Base):

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

    def __init__(self, data_dir='scannet', split='train', depth_transform='rgb', label_type='gt', is_subset=False, dataset_threshold=0.0) -> None:
        print(f"==> Loading {split} data from {data_dir}, Depth transform: {depth_transform}, Label type: {label_type} ({self})")
        split = 'valid' if split == 'test' else split
        super().__init__(data_dir, split, depth_transform, label_type, is_subset, dataset_threshold)
        return


    def _parse_file_list(self):

        split_meta = join(self.data_dir, f'{self.split}_every_100th.txt')
        with open(split_meta) as f:
            file_list = f.readlines()

        self.sample_stack = {}
        self.sample_stack['rgb'] = [join(self.data_dir, f'{self.split}/rgb', x.strip() + '.jpg') for x in file_list]
        self.sample_stack['depth'] = [join(self.data_dir, f'{self.split}/depth', x.strip() + '.png') for x in file_list]
        label_files = [join(self.data_dir, f'{self.split}/scene_class', x.strip()[:-6] + '.txt') for x in file_list]
        self.sample_stack['labels'] = []
        for label_file in label_files:
            with open(label_file) as f:
                self.sample_stack['labels'].append(self.class_name_to_id[remove_spaces_around_slash(f.readline())])

        assert len(self.sample_stack['rgb']) > 0, f"No rgb images found in {self.data_dir}/{self.split}/rgb"
        assert len(self.sample_stack['depth']) > 0, f"No depth images found in {self.data_dir}/{self.split}/depth"

        for i in range(len(self.sample_stack['rgb'])):
            assert splitext(basename(self.sample_stack['rgb'][i]))[0] == splitext(basename(self.sample_stack['depth'][i]))[0], f"RGB and depth images not matched: {self.sample_stack['rgb'][i]}, {self.sample_stack['depth'][i]}"


    def _parse_gt_id(self):
        split_meta = join(self.data_dir, f'{self.split}_every_100th.txt')
        with open(split_meta) as f:
            file_list = f.readlines()
        label_files = [join(self.data_dir, f'{self.split}/scene_class', x.strip()[:-6] + '.txt') for x in file_list]
        self.class_id_stack = []
        for label_file in label_files:
            with open(label_file) as f:
                self.class_id_stack.append(self.class_name_to_id[remove_spaces_around_slash(f.readline())])


    def _load_sample(self, index):
        input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_rgb = cv2.resize(input_rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
        input_depth = cv2.imread(self.sample_stack['depth'][index], -1)
        # Normalize depth to [0, 255]
        input_depth = ((input_depth - np.min(input_depth)) / (np.max(input_depth) - np.min(input_depth)) * 255.0).astype(np.uint8)

        return input_rgb, input_depth

    def __str__(self):
        return f"ScanNet v2"
