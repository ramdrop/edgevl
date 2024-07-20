#%%
from glob import glob
import numpy as np
from os.path import basename, dirname, exists, join, splitext

np.random.seed(0)
data_dir = 'xxx/EuroSAT_SWIR'
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
random_inds = np.random.choice(10, 10, replace=False)
#%%

train_files = []
for i in random_inds[:5]:
    scene_class = class_names[i]
    files = glob(join(data_dir, scene_class, '*.jpg'))
    files.sort()
    train_files += ['/'.join(file.split('/')[2:]) for file in files]
    print(scene_class, len(files))

test_files = []
for i in random_inds[5:]:
    scene_class = class_names[i]
    files = glob(join(data_dir, scene_class, '*.jpg'))
    files.sort()
    test_files += ['/'.join(file.split('/')[2:]) for file in files]
    print(scene_class, len(files))

with open(join('eurosat', 'train.txt'), 'w') as f:
    for file in train_files:
        f.write(file + '\n')

with open(join('eurosat', 'test.txt'), 'w') as f:
    for file in test_files:
        f.write(file + '\n')

print(f"train: {len(train_files)}, test: {len(test_files)}")
