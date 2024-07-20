#%%
from os.path import basename, dirname, exists, join, splitext
from glob import glob
import os
from tqdm import tqdm
import rasterio
import numpy as np
from PIL import Image

# order of bands in EuroSAT_MS: https://github.com/phelber/EuroSAT/issues/7
# "b1" , "b2" , "b3" , "b4", "b5","b6" , "b7" , "b8" , "b9" ,"b10" , "b11" , "b12", "b8a"
scene_class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
SRC = 'EuroSAT_MS'
DST = 'EuroSAT_SWIR'

for scene_class in scene_class_names:
    files = glob(join(SRC, scene_class, '*.tif'))
    if not exists(join(DST, scene_class)):
        os.makedirs(join(DST, scene_class))
    for file in tqdm(files):
        dst_dir = join(DST, scene_class)
        if not exists(dst_dir):
            os.makedirs(dst_dir)
        data = rasterio.open(file)
        img = data.read([12]) / 4095.            # "b12"
        img = img.squeeze()
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(join(dst_dir, basename(file)[:-4] + '.jpg'))
