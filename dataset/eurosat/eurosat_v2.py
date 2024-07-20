#%%
from glob import glob
from os.path import basename, dirname, exists, join, splitext
import cv2
import re
import sys
from dataset.base import Base


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


class EUROSAT(Base):
    class_id_to_name = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway', 4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'}
    class_name_to_id = {v: k for k, v in class_id_to_name.items()}
    clip_descriptions = [f"a satellite image of {determine_prefix(convert_string(x))}" for x in class_id_to_name.values()]

    def __init__(self, data_dir='eurosat', split='train', depth_transform='rgb', label_type='gt', is_subset=False, dataset_threshold=0.0) -> None:
        print(f"==> Loading {split} data from {data_dir}, Depth transform: {depth_transform}, Label type: {label_type} ({self})")
        super().__init__(data_dir, split, depth_transform, label_type, is_subset, dataset_threshold)
        return

    def _parse_file_list(self):
        # read from txt file
        files = []
        with open(join(self.data_dir, self.split + '.txt'), 'r') as f:
            for line in f:
                files.append(line.strip())
        files.sort()

        self.sample_stack = {}
        for modal in ['rgb', 'depth']:
            self.sample_stack[modal] = [join(self.data_dir, 'EuroSAT_RGB' if modal == 'rgb' else 'EuroSAT_SWIR', file) for file in files]
        for i in range(len(self.sample_stack['rgb'])):
            assert basename(self.sample_stack['rgb'][i]) == basename(self.sample_stack['depth'][i]), 'rgb and depth file name should be the same'


    def _parse_gt_id(self):
        self.class_id_stack = []
        for index in range(len(self.sample_stack['rgb'])):
            class_name = self.sample_stack['rgb'][index].split('/')[-2]
            self.class_id_stack.append(self.class_name_to_id[class_name])


    def _load_sample(self, index):
        input_rgb = cv2.cvtColor(cv2.imread(self.sample_stack['rgb'][index]), cv2.COLOR_BGR2RGB)
        input_depth = cv2.imread(self.sample_stack['depth'][index], cv2.IMREAD_UNCHANGED)

        return input_rgb, input_depth


    def __str__(self):
        return f"EUROSAT v2"
