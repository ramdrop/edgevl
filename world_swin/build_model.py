from world_swin import models
from utils.swin import load_checkpoint, load_pretrained
import torch

def build_model(config):
    model = models.build_model(config)
    return model


def load_weights(model, ckpt_path):
    print(f"Loading pretrained weights from {ckpt_path} ..")
    load_pretrained(ckpt_path, model)
    return

