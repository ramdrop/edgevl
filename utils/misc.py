import numpy as np
import random
import torch


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


import os
import subprocess


def schedule_device():
    process = subprocess.Popen("nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    info_per_card, error = process.communicate()
    process.kill()

    if error:
        print(f"Error retrieving GPU memory usage: {error}")
        return None

    info_per_card = info_per_card.decode('utf-8').strip().split('\n')
    if not info_per_card:
        print("Error: GPU memory usage information not found")
        return None

    card_memory_used = []
    for i in range(len(info_per_card)):
        if info_per_card[i] == '':
            continue
        else:
            total, used = map(int, info_per_card[i].split(','))
            card_memory_used.append(used)

    return card_memory_used.index(min(card_memory_used))


