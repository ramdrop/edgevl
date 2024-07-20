from dataset.nyu2.nyu2_v2 import NYU2
from dataset.sunrgbd.sunrgbd_v2 import SUNRGBD
from dataset.scannet.scannet_v2 import SCANNET
from dataset.eurosat.eurosat_v2 import EUROSAT

from dataset.nyu2.nyu2_ctrs_v2 import NYU2_CTRS
from dataset.sunrgbd.sunrgbd_ctrs_v2 import SUNRGBD_CTRS
from dataset.scannet.scannet_ctrs_v2 import SCANNET_CTRS
from dataset.eurosat.eurosat_ctrs_v2 import EUROSAT_CTRS



dataset = {
    'sunrgbd': SUNRGBD,
    'sunrgbd_ctrs': SUNRGBD_CTRS,
    'scannet': SCANNET,
    'scannet_ctrs': SCANNET_CTRS,
    'nyu2': NYU2,
    'nyu2_ctrs': NYU2_CTRS,
    'eurosat': EUROSAT,
    'eurosat_ctrs': EUROSAT_CTRS,
}

def get_dataset(name):
    return dataset[name]
