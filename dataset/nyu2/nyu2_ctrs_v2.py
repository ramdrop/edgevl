from dataset.nyu2.nyu2_v2 import NYU2
from dataset.base_ctrs import BaseCTRS

class NYU2_CTRS(BaseCTRS, NYU2):

    def __init__(self, data_dir='', split='train', depth_transform='rgb', label_type='pseudo_labels', is_subset=True, dataset_threshold=0.0, margin=0.1, n_neg=3, is_embedding_set=True, **kwargs) -> None:
        assert split in ['train'], f"Invalid split: {split}"
        super().__init__(data_dir=data_dir, split=split, depth_transform=depth_transform, label_type=label_type, is_subset=is_subset, dataset_threshold=dataset_threshold, margin=margin, n_neg=n_neg, is_embedding_set=is_embedding_set, **kwargs) # margin, n_neg, is_embedding_set
