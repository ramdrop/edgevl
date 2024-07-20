import torch
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(current_dir)

from models_dat_qm.dat import DAT

def build_model(config):
    model = DAT(**config['MODEL']['DAT'])
    return model

def load_weights(model, ckpt_path):
    print(f"==> Loading pretrained weights from {ckpt_path} ..")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    msg = model.load_pretrained(checkpoint['model'])
    print(f"==> {msg}")
    return

