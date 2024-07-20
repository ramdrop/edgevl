from dataset.base import Base
from os.path import join, splitext, basename
from glob import glob
import cv2
import numpy as np


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


class NYU2(Base):

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

    def __init__(self, data_dir='nyu2', split='train', depth_transform='rgb', label_type='gt', is_subset=False, dataset_threshold=0.0) -> None:
        super().__init__(data_dir, split, depth_transform, label_type, is_subset, dataset_threshold)
        return

    def _parse_file_list(self):
        # read from txt file
        idxs = []
        with open(join(self.data_dir, self.split + '.txt'), 'r') as f:
            for line in f:
                idxs.append(int(line.strip()))
        self.sample_stack = {}
        for modal in ['rgb', 'depth']:
            self.sample_stack[modal] = [join(self.data_dir, self.split, modal, f'{idx:04d}.png') for idx in idxs]

        for i in range(len(self.sample_stack['rgb'])):
            assert splitext(basename(self.sample_stack['rgb'][i]))[0] == splitext(basename(self.sample_stack['depth'][i]))[0], f"RGB and depth images not matched: {self.sample_stack['rgb'][i]}, {self.sample_stack['depth'][i]}"

        return

    def _parse_gt_id(self):
        self.class_id_stack = []
        # read from txt file
        idxs = []
        with open(join(self.data_dir, self.split + '.txt'), 'r') as f:
            for line in f:
                idxs.append(int(line.strip()))
        for idx in idxs:
            with open(join(self.data_dir, self.split, 'scene_class', f'{idx:04d}.txt'), 'r') as f:
                self.class_id_stack.append(self.class_name_to_id[f.read().strip()])


    def _load_sample(self, index):
        input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_depth = cv2.imread(self.sample_stack['depth'][index], cv2.IMREAD_UNCHANGED)
        return input_rgb, input_depth


    def _apply_transform(self, input_rgb, input_depth):
        transformed_image = self.shared_T(image=input_rgb, image_depth=input_depth)
        image_rgb = self.rgb_T(image=transformed_image['image'])

        input_depth = np.repeat(transformed_image['image_depth'][:, :, np.newaxis], repeats=3, axis=-1)
        input_depth = input_depth.astype(np.float32) / input_depth.max() * 255
        image_depth = self.depth_T(image=input_depth)

        image_depth_array = image_depth['image']
        image_rgb_array = image_rgb['image']
        image_rgb_array = np.transpose(image_rgb_array, (2, 0, 1))
        image_depth_array = np.transpose(image_depth_array, (2, 0, 1))
        return image_rgb_array, image_depth_array


    def __str__(self):
        return f"NYU2 v2"
