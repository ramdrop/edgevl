### Download the processed dataset
```
mkdir dbs && cd dbs

wget -O sunrgbd.zip "https://www.dropbox.com/scl/fi/t2lczsizzkgo1fcg7pn6o/sunrgbd.zip?rlkey=diisejj2hr6n94o03bzzhom05&dl=0"
unzip -q sunrgbd.zip
rm sunrgbd.zip

wget -O scannet.zip "https://www.dropbox.com/scl/fi/a34zmpssjn4zpavqj5m2i/scannet.zip?rlkey=te2zltcww6mn9hv1swl7ik6yv&dl=0"
unzip -q scannet.zip
rm scannet.zip

wget -O eurosat.zip "https://www.dropbox.com/scl/fi/dh0yc876n9ogu606b3e82/eurosat.zip?rlkey=pmvbfsdt8krovd7g3zaxihtlt&dl=0"
unzip -q eurosat.zip
rm eurosat.zip

wget -O nyu2.zip "https://www.dropbox.com/scl/fi/r8njk7u9u05vx1gf8luuj/nyu2.zip?rlkey=ymv74mb6hd18ho37fbd4nu4av&dl=0"
unzip -q nyu2.zip
rm nyu2.zip

```

### Or download the raw dataset and preprocess:

#### 1. NYU2

```
# Follow [https://github.com/TUI-NICR/nicr-scene-analysis-datasets/tree/main/nicr_scene_analysis_datasets/datasets/nyuv2]:

git clone git@github.com:TUI-NICR/nicr-scene-analysis-datasets.git

python -m pip install . [--user]

nicr_sa_prepare_dataset nyuv2 /path/where/to/store/nyuv2
```


#### 2. SUNRGBD
1. Download SUNRGBD dataset and toolkit from https://rgbd.cs.princeton.edu/challenge.html
2. The organized directory should be:
```
├── sunrgbd
│   ├── allsplit.mat
│   ├── organized-set
│   ├── SUNRGBD
│   └── SUNRGBDMeta.mat
```
3. git clone git@github.com:ramdrop/preprocess_sunrgbd.git
4. python main_steps.py --dataset-path ./sunrgbd/ --data-type crop --debug-mode 0

#### 3. ScanNet
```
(Follow [https://github.com/TUI-NICR/nicr-scene-analysis-datasets/blob/main/nicr_scene_analysis_datasets/datasets/scannet/README.md]:)

python download_scannet.py -o ./scannet_rgbd

nicr_sa_prepare_dataset scannet \
    /home/kaiwen/dataset/scannet_house/scannet_rgbd \
    /home/kaiwen/dataset/scannet_house/converted_train100_val100 \
    --n-processes 32 \
    --subsample 100 \
    --additional-subsamples 100

```

#### 4. EuroSAT
```
https://github.com/phelber/EuroSAT
- Download EuroSAT_MS.zip: https://zenodo.org/records/7711810#.ZAm3k-zMKEA
- Download EuroSAT_RGB.zip: https://zenodo.org/records/7711810#.ZAm3k-zMKEA

python extract_swir.py
python create_split.py

```
#### 5. Export CLIP pseudo labels
```
python export_clip_labels.py --dataset=[nyu2|sunrgbd|scannet|eurosat]
```