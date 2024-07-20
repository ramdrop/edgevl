from dataset.base import Base
from os.path import join, splitext, basename
from glob import glob


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


class SUNRGBD(Base):
    class_id_to_name = {0: "bathroom", 1: "bedroom", 2: "classroom", 3: "computer_room", 4: "conference_room", 5: "corridor", 6: "dining_area", 7: "dining_room", 8: "discussion_area", 9: "furniture_store", 10: "home_office", 11: "kitchen", 12: "lab", 13: "lecture_theatre", 14: "library", 15: "living_room", 16: "office", 17: "rest_space", 18: "study_space"}
    class_name_to_id = {v: k for k, v in class_id_to_name.items()}
    clip_descriptions = [f"a photo of {determine_prefix(convert_string(x))}" for x in class_id_to_name.values()]

    def __init__(self, data_dir='sunrgbd', split='train', depth_transform='rgb', label_type='gt', is_subset=False, dataset_threshold=0.0, **kwargs) -> None:
        super().__init__(data_dir=data_dir, split=split, depth_transform=depth_transform, label_type=label_type, is_subset=is_subset, dataset_threshold=dataset_threshold, **kwargs)
        return

    def _parse_file_list(self):
        self.sample_stack = {}
        for modal in ['rgb', 'depth']:
            self.sample_stack[modal] = glob(join(self.data_dir, self.split, modal, '*.jpg' if modal == 'rgb' else '*.png'))
            assert len(self.sample_stack[modal]) > 0, f"No {modal} images found in {self.data_dir}/{self.split}/{modal}"
            self.sample_stack[modal].sort()

        for i in range(len(self.sample_stack['rgb'])):
            assert splitext(basename(self.sample_stack['rgb'][i]))[0] == splitext(basename(self.sample_stack['depth'][i]))[0], f"RGB and depth images not matched: {self.sample_stack['rgb'][i]}, {self.sample_stack['depth'][i]}"

    def _parse_gt_id(self):
        self.class_id_stack = []
        for index in range(len(self.sample_stack['rgb'])):
            class_name = basename(self.sample_stack['rgb'][index]).split('__')[0]
            self.class_id_stack.append(self.class_name_to_id[class_name])

    def __str__(self):
        return f"SUNRGBD v2"



if __name__ == '__main__':
    train_dataset = SUNRGBD(data_dir='../dbs/sunrgbd', split='train', label_type='pseudo_labels')
    for i in range(5):
        print(train_dataset[i])
