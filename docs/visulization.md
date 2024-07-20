### Plot figures
1. Figure.1
```
visualization/feature_inspect_scannet/vis_openfig.py
```

2. Figure.3
```
visualization/feature_angles_scannet/feature_expain_v2.py
```

3. Figure.3 (TSNE)
```
visualization/feature_angles_scannet/feature_tsne.py
```

4. export clip features: clip-b
python evaluate_clip.py --dataset=eurosat --modal=rgb --clip_version=[G|B]


### Look into training parameters
1. Evaluate CLIP
```
python evaluate_clip.py  --dataset=[nyu2|sunrgbd|scannet|eurosat] --modal=[rgb|depth]  --clip_version=[B|G]
```


2. choose margin based on the feature distances before training
```
visualization/feature_distance/embedding_dist.py
```

3. Similarity
```
visualization/feature_distance/similarity_vs_acc1.py
```