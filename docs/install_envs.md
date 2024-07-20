#### 1. Configure Environment

```
conda create --name t2 python=3.10.13 -y
conda activate t2
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip3 install natten==0.14.6 -f https://shi-labs.com/natten/wheels/cu118/torch2.0.0/index.html 
pip install pytorch-quantization==2.1.2 --index-url https://pypi.nvidia.com 

```

#### 2. Download the pretrained weights

```
mkdir -p world_dat/pretrained_weights && cd pretrained_weights
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz
wget -O dat_pp_tiny_in1k_224.pth "https://www.dropbox.com/scl/fi/nwvi8ykcquuy30712icsy/dat_pp_tiny_in1k_224.pth?rlkey=m9wv2ky6wa1cey3lyvscf52im&dl=0"

mkdir -p world_swin/pretrained_weights && cd pretrained_weights
wget -O swin_tiny_patch4_window7_224.pth "https://www.dropbox.com/scl/fi/47awgzg0huiw5k1x0vho2/swin_tiny_patch4_window7_224.pth?rlkey=l47w1zqxhjvget20dnlup6stw&dl=0"

```

#### 3. Config
```
mkdir .wandb_config && cd .wandb_config && touch setup.yaml
# in setup.yaml file:
entity: [your username]
project: [your project name]
```