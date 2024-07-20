import pytorch_quantization.nn as quant_nn
from pytorch_quantization import calib
from tqdm import tqdm 


def collect_stats(model, data_loader, config, device):
    """Feed data to the network and collect statistic"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (rgb_imgs, depth_imgs, labels) in enumerate(tqdm(data_loader, total=config.calibration.num_batch, leave=False)):
        if i >= config.calibration.num_batch:
            break
        if config.calibration.modal == 'rgb':
            input_imgs = rgb_imgs.to(device)   # ([32, 4, 3, 224, 224]), ([32])
        elif config.calibration.modal == 'depth':
            input_imgs = depth_imgs.to(device) # ([32, 4, 3, 224, 224]), ([32])
        elif config.calibration.modal == 'rgbd':
            input_imgs = rgb_imgs.to(device)
            model(input_imgs.cuda())
            input_imgs = depth_imgs.to(device)
        model(input_imgs.cuda())

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, config, device):
    # Load calib result
    print_cnt = 2
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    if '_weight_quantizer' in name:
                        module.load_calib_amax(**config['weight_cal']['amax_method'])
                    else:
                        module.load_calib_amax(**config['act_cal']['amax_method'])
            if print_cnt > 0:
                print_cnt -= 1
                print(f"{name:40}: {module}")
    model.to(device)
