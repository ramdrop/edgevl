import os
# show current working directory
from world_vit.models.modeling_q import VisionTransformer, CONFIGS
import numpy as np
import torch 

def build_model(config):
    cfg = CONFIGS[config.MODEL.NAME]
    kwargs = {k: v for k, v in config.items() if 'quantization' in k}
    if config.CKPT == '':
        model = VisionTransformer(cfg, num_classes=config.MODEL.clip_dim, zero_head=True, img_size=224, vis=True, **kwargs)
    else:
        model = VisionTransformer(cfg, num_classes=config.MODEL.clip_dim, zero_head=False, img_size=224, vis=True, **kwargs)
    return model

def load_weights(model, ckpt_path):
    print(f"Loading pretrained weights from {ckpt_path} ..")
    if ckpt_path.endswith('.npz'):
        # load from npz
        model.load_from(np.load(ckpt_path))
    else:
        # load model weight
        state_dict = torch.load(ckpt_path, map_location='cpu')        
        # unsqueeze transformer.embeddings.patch_embeddings.bias
        if len(model.transformer.embeddings.patch_embeddings.bias.shape) > len(state_dict['model']["transformer.embeddings.patch_embeddings.bias"].shape):
            state_dict['model']["transformer.embeddings.patch_embeddings.bias"] = state_dict['model']["transformer.embeddings.patch_embeddings.bias"].unsqueeze(-1).unsqueeze(-1)
        elif len(model.transformer.embeddings.patch_embeddings.bias.shape) < len(state_dict['model']["transformer.embeddings.patch_embeddings.bias"].shape):
            state_dict['model']["transformer.embeddings.patch_embeddings.bias"] = state_dict['model']["transformer.embeddings.patch_embeddings.bias"].squeeze()
        print(state_dict.keys())
        # load state_dict
        model.load_state_dict(state_dict['model'], strict=False)