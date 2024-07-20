import argparse
from easydict import EasyDict as edict
import ruamel.yaml
import yaml
import wandb
from os.path import join, exists, basename
import os
import sys
import shlex
from utils.misc import schedule_device
os.environ['WANDB_SILENT'] = 'true'


def request_from_wandb(run_name, wandb_project):
    run_dir = join('logs', run_name, 'wandb/latest-run/files')
    if not exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
        print(f'{run_dir} does not exist, trying to download from wandb ...')
        api = wandb.Api()
        runs = api.runs(wandb_project)
        for run in runs:
            if run.name == args.run_name:
                wandb_run_id = run.id
        run = api.run(f'{wandb_project}/{wandb_run_id}')
        print(f"Downloading weights from wandb ...")
        best_model = wandb.restore('src/best_model.pth', run_path=f"{wandb_project}/{wandb_run_id}", replace=True, root=run_dir)
        print(f"Downloading global config from wandb ...")
        wandb_global_config_f = wandb.restore('config.yaml', run_path=f"{wandb_project}/{wandb_run_id}", replace=True, root=run_dir)
        with open(wandb_global_config_f.name, 'r') as stream:
            wandb_global_config = yaml.safe_load(stream)
        print(f"Downloading training config from wandb ...")
        restored_config = wandb.restore(f"src/{wandb_global_config['CONFIG']['value']}", run_path=f"{wandb_project}/{wandb_run_id}", replace=True, root=run_dir)
        print(f"restored config: {restored_config.name}")

        return restored_config.name, best_model.name
    else:
        return join(run_dir, 'src', 'config.yaml'), join(run_dir, 'src', 'best_model.pth')

def parse_world(run_name, config):
    infer_pattern = ''
    if run_name != '':
        infer_pattern = run_name
    if config != '':
        infer_pattern = basename(config).split('_')[0]

    if infer_pattern.startswith('s'):
        world = 'swin'
    elif infer_pattern.startswith('d'):
        world = 'dat'
    elif infer_pattern.startswith('v'):
        world = 'vit'
    else:
        raise ValueError("Invalid run_name. It should start with either 's', 'd' or 'v'")

    return world

if __name__ == '__main__':
    # Reconstruct the launch command
    launch_command = ' '.join(shlex.quote(arg) for arg in sys.argv)
    print(f"Launch command: python {launch_command}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, choices=[0, 1, 2, 3], help="which GPU to use")
    parser.add_argument('--offline', action='store_true', help='whether to upload to wandb')
    parser.add_argument('--phase', type=str, default='', choices=['train', 'train_ctrs', 'test', 'train_teacher', 'train_student', 'train_ctrs_kd'], help='path to config') # make sure args.phase==config.PHASE before running
    parser.add_argument('--world', type=str, default='', help='world name')
    parser.add_argument('--config', type=str, default='', help='path to config')
    parser.add_argument('--quant_config', type=str, default='', help='path to quantization config')
    parser.add_argument('--trainer', type=str, default='trainer', help='trainer class name')
    parser.add_argument('--pre_calibration', action='store_true', help='whether to pre-calibrate')
    # for test
    parser.add_argument('--run_name', type=str, default='', help='run name')
    parser.add_argument('--test_modal', type=str, default='', choices=['rgb', 'depth'], help='set modal to evaluate')
    parser.add_argument('--test_dataset', type=str, default='', help='')
    parser.add_argument('--static_or_dynamic', type=str, default='dynamic', help='static or dynamic quantization')
    parser.add_argument('--dataset', type=str, default='', help='dataset') # in case we want to test on a different dataset
    # embedding profile
    parser.add_argument('--embedding_profile', action='store_true', help='whether to profile the embedding')
    parser.add_argument('--export_features', action='store_true', help='whether to profile the embedding')

    args = parser.parse_args()

    # set gpu
    if args.gpu == -1:
        args.gpu = schedule_device()

    # read wandb settings
    assert exists('.wandb_config/setup.yaml'), "Please configure your wandb settings first."
    with open('.wandb_config/setup.yaml', 'r') as stream:
        wandb_config = yaml.safe_load(stream)
    wandb_project = f"{wandb_config['entity']}/{wandb_config['project']}"

    # identify which backbone we are using from the run_name or config filename
    args.world = parse_world(args.run_name, args.config)
    args.config = join(f"world_{args.world}", args.config)

    # training phase
    if args.phase in ['train', 'train_ctrs']:
        # load training parameters
        with open(args.config, 'r') as stream:
            config = edict(ruamel.yaml.safe_load(stream))
            assert config.CONFIG == basename(args.config), f"==> {args.config}:CONFIG != {basename(args.config)}"
            config.PHASE = args.phase
        # load quantization parameters
        with open(args.quant_config, 'r') as stream:
            config_quant = edict(ruamel.yaml.safe_load(stream))
        if config.MODEL.TYPE == 'dat':
            config.MODEL.DAT.update(config_quant)
        elif config.MODEL.TYPE in ['swin', 'vit']:
            config.update(config_quant)
            config.quantization.pre_calibration = args.pre_calibration
        else:
            raise ValueError(f"Model type {config.MODEL.TYPE} not supported")
        # resume from a trained model
        if args.run_name != '':
            restored_config_path, best_model_path = request_from_wandb(args.run_name, wandb_project)
            config.CKPT = best_model_path
            
    # testing phase
    elif args.phase in ['test']:
        run_dir = join('logs', args.run_name, 'wandb/latest-run/files')
        if exists(run_dir):
            with open(join(run_dir, 'config.yaml'), 'r') as stream:
                wandb_global_config = yaml.safe_load(stream)
                restored_config = join(run_dir, 'src', wandb_global_config['CONFIG']['value'])
            print(f"==> Restored config: {restored_config}")
            with open(restored_config, 'r') as stream:
                config = edict(ruamel.yaml.safe_load(stream))
                config.CKPT = join(run_dir, 'src', 'best_model.pth')
        else:
            restored_config_path, best_model_path = request_from_wandb(args.run_name, wandb_project)
            with open(restored_config_path, 'r') as stream:
                config = edict(ruamel.yaml.safe_load(stream))
                config.CKPT = best_model_path

        config.PHASE = 'test'
        if args.test_modal != '':
            config.MODAL = args.test_modal
        if args.test_dataset != '':
            config.DATA.DATASET = args.test_dataset

        with open(args.quant_config, 'r') as stream:
            config_quant = edict(ruamel.yaml.safe_load(stream))
            if config.MODEL.TYPE == 'dat':
                config.MODEL.DAT.update(config_quant)
            elif config.MODEL.TYPE in ['swin', 'vit']:
                config.update(config_quant)
            else:
                raise ValueError(f"Model type {config.MODEL.TYPE} not supported")

    else:
        raise ValueError(f"Phase {args.phase} not supported")

    print(f"==> {config_quant}")
    trainer_cls = getattr(__import__(args.trainer), 'Trainer')
    config.update({'cmd': launch_command})
    trainer = trainer_cls(config, args)
    trainer.launch_command = launch_command
    assert config.PHASE in ['train', 'train_ctrs', 'test']
    print(f"==> Trainer: {trainer_cls}")
    print(f"==> Phase: {config.PHASE} ...")

    if config.PHASE in ['train', 'train_ctrs']:
        trainer.train()
    elif config.PHASE in ['test']:
        trainer.test(args.static_or_dynamic, config_quant)
    else:
        raise ValueError(f"Phase {config.PHASE} not supported")
