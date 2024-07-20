import sys
sys.path.append('../')
import os
from os.path import join, dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorrt as trt
from dataset.scannet.scannet_v2 import SCANNET
import warnings

# Ignore DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
# arg parser
import argparse

class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine context creating
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            # set shape 
            self.context.set_input_shape(input_name, tuple(inputs[i].shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_tensor_shape(output_name))
            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs
    
def evaluate(model, modal, data_loader_val, text_features, ImgNum):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnt_correct = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.Event(enable_timing=True)
    time = 0
    for batch_idx, (rgb_imgs, depth_imgs, class_id) in enumerate(tqdm(data_loader_val, total=ImgNum,leave=False)):
        # print(class_id)
        if modal == 'rgb':
            input_imgs = rgb_imgs.to(device)   # ([32, 4, 3, 224, 224]), ([32])
        elif modal == 'depth':
            input_imgs = depth_imgs.to(device) # ([32, 4, 3, 224, 224]), ([32])
        else:
            raise NotImplementedError

        with torch.no_grad():
            starter.record()
            image_features = model(input_imgs) # ([1, 512])
            ender.record()
            torch.cuda.synchronize()
            time  += starter.elapsed_time(ender)
            
        # Pick the top 5 most similar labels for the image
        image_features = F.normalize(image_features, p=2, dim=-1)
        similarity = (100.0 * image_features @ text_features.float().T).softmax(dim=-1) # ([1, 19])

        for i in range(len(similarity)):
            values, indices = similarity[i].topk(5)
            # print(indices[0].item(), class_id[i].item())
            if indices[0].item() == class_id[i].item():
                cnt_correct += 1
        if batch_idx >= ImgNum:
            break
    acc1 = cnt_correct / ImgNum

    print(f'=> GPU Latency: {time:.2f} ms, FPS: {ImgNum*1000/time:.2f}')
    print(f'=> Modality {modal} Accuracy: {acc1:.2f}')

def build_model(model_name):
    f = open(f'onnx/engines/{model_name}.trt', "rb")
    trt.init_libnvinfer_plugins(None, "")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    names = []
    for idx in range(engine.num_bindings):
        name = engine.get_tensor_name(idx)
        is_input = engine.get_tensor_mode(name)
        op_type = engine.get_tensor_dtype(name)
        # model_all_names.append(name)
        shape = engine.get_tensor_shape(name)
        names.append(name)
        print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)
    trt_module = TRTModule(engine, [names[0]], [names[1]])
    return trt_module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='clip_Big',help='model')
    parser.add_argument('--modality', default='rgb',help='modality')
    # number of image
    parser.add_argument('--num_img', default=1000, type=int, help='number of images')
    args = parser.parse_args()

    dataset_val = SCANNET(split='test', data_dir=join('../dbs', 'scannet'), depth_transform="rgb", label_type='gt')
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model != 'clip_Big':
        text_features = torch.load('../_cache/text_features_scannet.pt', map_location='cpu').to(device)
    else:
        text_features = torch.load('../_cache/text_features_clip_scannet.pt',  map_location='cpu').to(device)

    model = build_model(args.model)
    evaluate(model, args.modality, data_loader_val, text_features, args.num_img)

if __name__ == '__main__':
    main()