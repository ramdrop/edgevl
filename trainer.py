import sys
sys.path.append('../')
import os

import yaml
import datetime
import shutil
from os.path import join, dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import rich
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_quantization.nn as quant_nn
import open_clip
import ipdb
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.optim as optim

from dataset.build_dataset import get_dataset
from utils.misc import setup_seed, get_grad_norm
from quantization_libs.calibrator import collect_stats, compute_amax
import importlib


class Trainer:
    def __init__(self, config, args):
        setup_seed(config.SEED)
        launch_command = None
        build_scheduler_fn = getattr(importlib.import_module(f"world_{args.world}.lr_scheduler"), 'build_scheduler')
        build_optimizer_fn = getattr(importlib.import_module(f"world_{args.world}.optimizer"), 'build_optimizer')
        build_model_fn = getattr(importlib.import_module(f"world_{args.world}.build_model"), 'build_model')
        load_weights_fn = getattr(importlib.import_module(f"world_{args.world}.build_model"), 'load_weights')

        # BUILD RUN DIR ====================== #
        timestamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
        if config.PHASE in ['train', 'train_ctrs']:
            cur_name = f"{args.config.split('/')[-1].split('_')[0]}_{timestamp}" # configs/sunrgbd/datt_depth.yaml
            self.run_dir = os.path.join('logs', cur_name)
            os.makedirs(self.run_dir, exist_ok=True)
            with open(join(self.run_dir, 'config.yaml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            with open('.wandb_config/setup.yaml', 'r') as stream:
                wandb_config = yaml.safe_load(stream)
                wandb_project = f"{wandb_config['entity']}/{wandb_config['project']}"                
            wandb.init(project=wandb_project, save_code=True, config=config, dir=self.run_dir, name=cur_name, mode='offline' if args.offline else 'online')
            self.wandb_dir = join(wandb.run.dir, 'src')
            os.makedirs(self.wandb_dir, exist_ok=True)
            for file in ["run.py", os.path.abspath(__file__), args.config, args.quant_config]:
                shutil.copy(file, self.wandb_dir)
            shutil.copytree("dataset", join(self.wandb_dir, 'dataset'), ignore=shutil.ignore_patterns('__pycache__'))
            wandb.save(join(self.wandb_dir, "*"), base_path=dirname(self.wandb_dir), policy="live")
        elif config.PHASE in ['test']:
            self.run_dir = join('logs', args.run_name)
        else:
            raise ValueError(f"Phase {config.PHASE} not supported")

        # self.device
        torch.cuda.set_device(args.gpu)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.clip_dim
        if config.MODEL.TYPE == 'dat':
            self.clip_dim = config.MODEL.DAT.clip_dim
        elif config.MODEL.TYPE == 'swin':
            self.clip_dim = config.MODEL.SWIN.clip_dim
        elif config.MODEL.TYPE in ['vit']:
            self.clip_dim = config.MODEL.clip_dim
        else:
            raise ValueError(f"Model type {config.MODEL.TYPE} not supported")

        # LOAD DATASET ======================= #
        assert config.DATA.DATASET in ['sunrgbd', 'scannet', 'nyu2', 'eurosat'], f"Dataset {config.DATA.DATASET} not supported"
        if config.PHASE in ['train', 'test']:
            self.dataset_train = get_dataset(config.DATA.DATASET)(
                split='train',
                data_dir=join(config.DATA.ROOT, config.DATA.DATASET),
                depth_transform=config.DATA.DEPTH_TRANSFORM,
                label_type=config.DATA.LABEL_TYPE,
                is_subset=getattr(config.DATA, 'IS_SUBSET', False),
                dataset_threshold=getattr(config.DATA, 'DATASET_THRESHOLD', 0.0),
            )

            config.DATA.BATCH_SIZE = config.DATA.VAL_BATCH_SIZE if config.PHASE == 'test' else config.DATA.BATCH_SIZE
            self.data_loader_train = DataLoader(self.dataset_train, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
        elif config.PHASE in ['train_ctrs']:
            self.embedding_dataset = get_dataset(f"{config.DATA.DATASET}_ctrs")(
                split='train',
                data_dir=join(config.DATA.ROOT, config.DATA.DATASET),
                depth_transform=config.DATA.DEPTH_TRANSFORM,
                label_type=config.DATA.LABEL_TYPE,
                is_subset=config.DATA.IS_SUBSET,
                dataset_threshold=config.DATA.DATASET_THRESHOLD,
                margin=config.TRAIN.CRETERION.TRIPLET_MARGIN,
                n_neg=config.TRAIN.CRETERION.NEG_NUM,
                is_embedding_set=True,
                mining_method=getattr(config.TRAIN.CRETERION, 'MINING_METHOD', {}),
            )
            self.dataset_train = get_dataset(f"{config.DATA.DATASET}_ctrs")(
                split='train',
                data_dir=join(config.DATA.ROOT, config.DATA.DATASET),
                depth_transform=config.DATA.DEPTH_TRANSFORM,
                label_type=config.DATA.LABEL_TYPE,
                is_subset=config.DATA.IS_SUBSET,
                dataset_threshold=config.DATA.DATASET_THRESHOLD,
                margin=config.TRAIN.CRETERION.TRIPLET_MARGIN,
                n_neg=config.TRAIN.CRETERION.NEG_NUM,
                is_embedding_set=False,
                mining_method=getattr(config.TRAIN.CRETERION, 'MINING_METHOD', {}),
            )
            self.embedding_data_loader = DataLoader(self.embedding_dataset, batch_size=config.DATA.EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
            self.data_loader_train = DataLoader(self.dataset_train, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=False, collate_fn=self.dataset_train.collate_fn)
        else:
            raise ValueError(f'{config.PHASE} not supported')

        self.dataset_val = get_dataset(config.DATA.DATASET)(
            split='test',
            data_dir=join(config.DATA.ROOT, config.DATA.DATASET),
            depth_transform=config.DATA.DEPTH_TRANSFORM,
            label_type='gt',
        )
        self.data_loader_val = DataLoader(self.dataset_val, batch_size=config.DATA.VAL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        print(f"==> Dataset: {config.DATA.DATASET}, Train set: {len(self.dataset_train)}, Batch size: {config.DATA.BATCH_SIZE}, Val set: {len(self.dataset_val)}, Batch size: {config.DATA.VAL_BATCH_SIZE}")

        # LOAD MODEL ========================= #
        self.model = build_model_fn(config)
        self.model = self.model.to(self.device)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if config.PHASE in ['train']:
            if config.CKPT != '':
                ckpt_path = join(current_dir, config.CKPT)
            else:
                ckpt_path = join(current_dir, f"world_{args.world}", config.MODEL.PRETRAINED)
            load_weights_fn(self.model, ckpt_path)
            self.optimizer = build_optimizer_fn(config, self.model)
            self.lr_scheduler = build_scheduler_fn(config, self.optimizer, len(self.data_loader_train))
        elif config.PHASE in ['test']:
            load_weights_fn(self.model, join(current_dir, config.CKPT))
        elif config.PHASE in ['train_ctrs']:
            load_weights_fn(self.model, join(current_dir, config.CKPT))
            self.optimizer = build_optimizer_fn(config, self.model)
            self.lr_scheduler = build_scheduler_fn(config, self.optimizer, len(self.data_loader_train))
        else:
            raise ValueError(f'{config.PHASE} not supported')

        # LOSS FUNCTION ====================== #
        if config.PHASE in ['train', 'train_ctrs']:
            print(f"==> Loss function: {config.TRAIN.CRETERION.NAME}")
            print(f"==> Mix config: {getattr(config, 'MIX_INPUT', None)}")
            if config.TRAIN.CRETERION.NAME == 'L1':
                self.criterion = nn.L1Loss(reduction='none')
            elif config.TRAIN.CRETERION.NAME == 'MSE':
                self.criterion = nn.MSELoss(reduction='none')
            elif config.TRAIN.CRETERION.NAME == 'TripletMarginLoss':
                self.criterion = nn.TripletMarginLoss(margin=config.TRAIN.CRETERION.TRIPLET_MARGIN, p=2, reduction='mean').to(self.device)
            else:
                raise ValueError(f'{config.TRAIN.CRETERION.NAME} not supported')

        # LOAD CLIP ========================== #
        clip_modal_name = (config.CLIP_MODEL.NAME, config.CLIP_MODEL.PRETRAINED)
        self.model_clip, _, preprocess = open_clip.create_model_and_transforms(config.CLIP_MODEL.NAME, pretrained=config.CLIP_MODEL.PRETRAINED, device=self.device, cache_dir='_cache')
        tokenizer = open_clip.get_tokenizer(clip_modal_name[0])
        text_inputs = torch.cat([tokenizer(x) for x in self.dataset_val.clip_descriptions]).to(self.device)
        with torch.no_grad():
            text_features = self.model_clip.encode_text(text_inputs)                                       
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print(f"==> CLIP model: {clip_modal_name[0]}, pretrained: {clip_modal_name[1]}")

        if config.PHASE in ['train_ctrs']:
            kd_config = getattr(config.TRAIN.CRETERION, 'KD', None)
            if getattr(kd_config, 'ENABLE', False) != True:
                del self.model_clip
                torch.cuda.empty_cache()
                print(f"==> Remove clip model from memory.")
            else:
                print(f"==> KD config: {kd_config}")

        self.config = config
        self.args = args


    def train(self):
        acc1 = self.evaluate(-1)
        print(f"==> Epoch: -1, Accuracy: {acc1:.4f}")

        max_acc1 = 0
        for epoch in range(self.config.TRAIN.EPOCHS):
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.config.PHASE in ['train']:
                log_items = self.train_one_epoch(epoch)
            elif self.config.PHASE in ['train_ctrs']:
                self.build_embedding()
                if self.args.embedding_profile == True:
                    print("==> Profiling pair distance ...")
                    self.profile_pair_distance()
                    wandb.finish()
                    return
                log_items = self.train_ctrs_one_epoch(epoch)
            else:
                raise NotImplementedError

            acc1 = self.evaluate(epoch)
            print(f"==> Epoch: {epoch}, LR: {current_lr:.10f}, Accuracy: {acc1:.4f}")

            if acc1 > max_acc1:
                max_acc1 = acc1
                torch.save({'model': self.model.state_dict()}, join(self.wandb_dir, 'best_model.pth'))

            log_items['acc1'] = acc1
            log_items['max_acc1'] = max_acc1
            log_items['lr'] = self.optimizer.param_groups[0]['lr']
            log_items['epoch'] = epoch
            wandb.log(log_items)
        wandb.save(join(self.wandb_dir, "best_model.pth"), base_path=dirname(self.wandb_dir), policy="now")


    def build_embedding(self):
        cache = torch.zeros((len(self.embedding_dataset), self.clip_dim), device=self.device) 
        with torch.no_grad():
            for iteration, (rgb, depth, indices) in enumerate(tqdm(self.embedding_data_loader, leave=False), 1):
                rgb = rgb.to(self.device)                                                    
                emb = self.model(rgb)                                                         
                cache[indices, :] = emb
                del rgb, emb
        self.dataset_train.embedding_cache = cache.to(torch.device("cpu"))

    def profile_pair_distance(self):
        nums_negs, rgb_positive_dists, rgb_negative_dists, depth_positive_dists, depth_negative_dists = [], [], [], [], []

        for idx, (imgs, nums_neg, labels) in enumerate(tqdm(self.data_loader_train, leave=False)):
            if imgs is None:
                continue

            B = len(nums_neg)
            nums_neg_total = torch.sum(nums_neg)     
            nums_negs.append(nums_neg_total)
            input_imgs = imgs.to(self.device)        
            with torch.no_grad():
                features_dat = self.model(input_imgs)
            features_dat = F.normalize(features_dat, p=2, dim=-1)

            a_rgb_emb, p_rgb_emb, n_rgb_emb, a_depth_emb, p_depth_emb, n_depth_emb = torch.split(features_dat, [B, B, nums_neg_total, B, B, nums_neg_total])
            loss_rgb_triplet = 0
            loss_depth_triplet = 0
            for i, num_neg in enumerate(nums_neg):
                for n in range(num_neg):
                    negIx = (torch.sum(nums_neg[:i]) + n).item()
                    loss_rgb_triplet += self.criterion(a_rgb_emb[i:i + 1], p_rgb_emb[i:i + 1], n_rgb_emb[negIx:negIx + 1])
                    loss_depth_triplet += self.criterion(a_depth_emb[i:i + 1], p_depth_emb[i:i + 1], n_depth_emb[negIx:negIx + 1])

                    rgb_positive_dists.append(F.pairwise_distance(a_rgb_emb[i:i + 1], p_rgb_emb[i:i + 1], p=2).item())
                    rgb_negative_dists.append(F.pairwise_distance(a_rgb_emb[i:i + 1], n_rgb_emb[negIx:negIx + 1], p=2).item())
                    depth_positive_dists.append(F.pairwise_distance(a_depth_emb[i:i + 1], p_depth_emb[i:i + 1], p=2).item())
                    depth_negative_dists.append(F.pairwise_distance(a_depth_emb[i:i + 1], n_depth_emb[negIx:negIx + 1], p=2).item())

        rgb_positive_dists, rgb_negative_dists, depth_positive_dists, depth_negative_dists, nums_negs = map(torch.tensor, [rgb_positive_dists, rgb_negative_dists, depth_positive_dists, depth_negative_dists, nums_negs])
        # save to disk
        torch.save({'rgb_positive_dists': rgb_positive_dists, 'rgb_negative_dists': rgb_negative_dists, 'depth_positive_dists': depth_positive_dists, 'depth_negative_dists': depth_negative_dists, 'nums_negs': nums_negs}, join(self.wandb_dir, 'pair_distance.pth'))
        print(f"==> Pair distance profile saved to {join(self.wandb_dir, 'pair_distance.pth')}")

    def train_ctrs_one_epoch(self, epoch):
        log_items = {}
        num_steps = len(self.data_loader_train)
        nums_negs, rgb_positive_dists, rgb_negative_dists, depth_positive_dists, depth_negative_dists = [], [], [], [], []
        for idx, (imgs, nums_neg, labels) in enumerate(tqdm(self.data_loader_train, leave=False)):
            if imgs is None:
                continue

            B = len(nums_neg)
            nums_neg_total = torch.sum(nums_neg)  
            nums_negs.append(nums_neg_total.float().item() / B)
            log_items['nums_neg_avg'] = nums_neg_total.float().item() / B
            input_imgs = imgs.to(self.device)    
            features_stu = self.model(input_imgs) 
            features_stu = F.normalize(features_stu, p=2, dim=-1)

            a_rgb_emb, p_rgb_emb, n_rgb_emb, a_depth_emb, p_depth_emb, n_depth_emb = torch.split(features_stu, [B, B, nums_neg_total, B, B, nums_neg_total])
            loss_rgb_triplet = 0
            loss_depth_triplet = 0
            for i, num_neg in enumerate(nums_neg):
                for n in range(num_neg):
                    negIx = (torch.sum(nums_neg[:i]) + n).item()
                    loss_rgb_triplet += self.criterion(a_rgb_emb[i:i + 1], p_rgb_emb[i:i + 1], n_rgb_emb[negIx:negIx + 1])
                    loss_depth_triplet += self.criterion(a_depth_emb[i:i + 1], p_depth_emb[i:i + 1], n_depth_emb[negIx:negIx + 1])
                    rgb_positive_dists.append(F.pairwise_distance(a_rgb_emb[i:i + 1], p_rgb_emb[i:i + 1], p=2).item())
                    rgb_negative_dists.append(F.pairwise_distance(a_rgb_emb[i:i + 1], n_rgb_emb[negIx:negIx + 1], p=2).item())
                    depth_positive_dists.append(F.pairwise_distance(a_depth_emb[i:i + 1], p_depth_emb[i:i + 1], p=2).item())
                    depth_negative_dists.append(F.pairwise_distance(a_depth_emb[i:i + 1], n_depth_emb[negIx:negIx + 1], p=2).item())

            loss_rgb_triplet /= nums_neg_total.float()   # normalise by actual number of negatives
            loss_depth_triplet /= nums_neg_total.float() # normalise by actual number of negatives
            loss = loss_rgb_triplet + loss_depth_triplet

            log_items['loss_rgb_triplet'] = loss_rgb_triplet.item()
            log_items['loss_depth_triplet'] = loss_depth_triplet.item()
            log_items['loss'] = loss.item()

            loss.backward()
            if self.config.TRAIN.CLIP_GRAD:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(self.model.parameters())
            log_items['grad_norm'] = grad_norm
            self.optimizer.step()

            if isinstance(self.lr_scheduler, optim.lr_scheduler.ExponentialLR):
                pass
            elif isinstance(self.lr_scheduler, CosineLRScheduler):
                self.lr_scheduler.step_update(epoch * num_steps + idx)

        if isinstance(self.lr_scheduler, optim.lr_scheduler.ExponentialLR):
            self.lr_scheduler.step()
        elif isinstance(self.lr_scheduler, CosineLRScheduler):
            pass
        else:
            raise NotImplementedError

        rgb_positive_dists, rgb_negative_dists, depth_positive_dists, depth_negative_dists, nums_negs = map(torch.tensor, [rgb_positive_dists, rgb_negative_dists, depth_positive_dists, depth_negative_dists, nums_negs])
        log_items['emb/rgb_pos_mean'] = rgb_positive_dists.mean().item()
        log_items['emb/rgb_neg_mean'] = rgb_negative_dists.mean().item()
        log_items['emb/rgb_pos_neg_mean'] = (rgb_positive_dists - rgb_negative_dists).mean().item()
        log_items['emb/depth_pos_mean'] = depth_positive_dists.mean().item()
        log_items['emb/depth_neg_mean'] = depth_negative_dists.mean().item()
        log_items['emb/depth_pos_neg_mean'] = (depth_positive_dists - depth_negative_dists).mean().item()
        log_items['emb/nums_neg_mean'] = nums_negs.float().mean().item()

        return log_items


    def process_input_images(self, rgb_imgs, depth_imgs, modal, mix_input_config):
        if getattr(mix_input_config, 'ENABLE', False) == True:
            input_imgs = torch.cat([depth_imgs, rgb_imgs], dim=0) 
        else:
            if modal == 'rgb':
                input_imgs = rgb_imgs
            elif modal == 'depth':
                input_imgs = depth_imgs
            else:
                raise NotImplementedError

        features = self.model(input_imgs)        
        features = F.normalize(features, p=2, dim=-1)
        return features


    def compute_loss(self, features_stu, features_clip, mix_input_config):
        if getattr(mix_input_config, 'ENABLE', False) == True:
            features_clip = features_clip.unsqueeze(0).repeat(2, 1, 1).view(-1, features_clip.shape[-1]) 
            loss = self.criterion(features_stu, features_clip).mean()                              

        else:
            loss = self.criterion(features_stu, features_clip).mean()                          

        return loss


    def train_one_epoch(self, epoch):
        log_items = {}
        num_steps = len(self.data_loader_train)

        modal = self.config.MODAL
        mix_input_config = getattr(self.config, 'MIX_INPUT', None)
        for idx, (rgb_imgs, depth_imgs, labels) in enumerate(tqdm(self.data_loader_train, leave=False)):
            self.model.train()
            self.optimizer.zero_grad()
            rgb_imgs, depth_imgs = rgb_imgs.to(self.device), depth_imgs.to(self.device)

            with torch.no_grad():
                features_clip = self.model_clip.encode_image(rgb_imgs)
                features_clip /= features_clip.norm(dim=-1, keepdim=True)

            features_stu = self.process_input_images(rgb_imgs, depth_imgs, modal, mix_input_config)
            loss = self.compute_loss(features_stu, features_clip, mix_input_config, log_items)

            log_items['loss'] = loss.item()

            loss.backward()

            if self.config.TRAIN.CLIP_GRAD:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(self.model.parameters())

            log_items['grad_norm'] = grad_norm
            self.optimizer.step()

            if isinstance(self.lr_scheduler, optim.lr_scheduler.ExponentialLR):
                pass
            elif isinstance(self.lr_scheduler, CosineLRScheduler):
                self.lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                raise NotImplementedError

        if isinstance(self.lr_scheduler, optim.lr_scheduler.ExponentialLR):
            self.lr_scheduler.step()
        elif isinstance(self.lr_scheduler, CosineLRScheduler):
            pass
        else:
            raise NotImplementedError

        return log_items


    def evaluate(self, epoch, export_features=False):
        self.model.eval()
        cnt_correct = 0
        if export_features:
            features = np.zeros((len(self.data_loader_val.dataset), self.clip_dim), dtype=np.float32)
            labels = np.zeros((len(self.data_loader_val.dataset)), dtype=np.int32)
        for batch_idx, (rgb_imgs, depth_imgs, class_id) in enumerate(tqdm(self.data_loader_val, leave=False)):
            batch_size = rgb_imgs.shape[0]
            batch_diff = self.config.DATA.VAL_BATCH_SIZE - batch_size

            # padd tensors with 0s to make sure they have the same size
            if rgb_imgs.shape[0] < self.config.DATA.VAL_BATCH_SIZE:
                rgb_imgs = torch.cat([rgb_imgs, torch.zeros([batch_diff] + list(rgb_imgs.shape)[1:])], dim=0)
                depth_imgs = torch.cat([depth_imgs, torch.zeros([batch_diff] + list(depth_imgs.shape)[1:])], dim=0)
                class_id = torch.cat([class_id, torch.zeros([batch_diff] + list(class_id.shape)[1:])], dim=0)

            if self.config.MODAL == 'rgb':
                input_imgs = rgb_imgs.to(self.device)
            elif self.config.MODAL == 'depth':
                input_imgs = depth_imgs.to(self.device)
            else:
                raise NotImplementedError

            with torch.no_grad():
                image_features = self.model(input_imgs)

            image_features = F.normalize(image_features, p=2, dim=-1)
            similarity = (100.0 * image_features @ self.text_features.float().T).softmax(dim=-1)
            for i in range(len(similarity)):
                if i >= batch_size:
                    break
                values, indices = similarity[i].topk(5)
                if indices[0].item() == class_id[i].item():
                    cnt_correct += 1
            if export_features:
                features[batch_idx * self.config.DATA.VAL_BATCH_SIZE:batch_idx * self.config.DATA.VAL_BATCH_SIZE + batch_size] = image_features[:batch_size].cpu().numpy()
                labels[batch_idx * self.config.DATA.VAL_BATCH_SIZE:batch_idx * self.config.DATA.VAL_BATCH_SIZE + batch_size] = class_id[:batch_size].cpu().numpy()

        acc1 = cnt_correct / len(self.data_loader_val.dataset)
        if export_features:
            np.save(join(self.run_dir, f'features_{self.config.MODAL}.npy'), features)
            np.save(join(self.run_dir, f'labels_{self.config.MODAL}.npy'), labels)
            print(f"==> Features saved to {join(self.run_dir, f'features_{self.config.MODAL}.npy')}")

        return acc1


    def test(self, static_or_dynamic, config_quant):
        if static_or_dynamic == 'static':
            assert config_quant.quantization.method == 'jacob', 'Only support static quantization with jacob method'
            with torch.no_grad():
                collect_stats(self.model, self.data_loader_train, config_quant.quantization, self.device)
                compute_amax(self.model, config_quant.quantization, self.device)

        print_cnt = 2
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if '_weight_quantizer' in name or '_input_quantizer' in name:
                    if print_cnt > 0:
                        print_cnt -= 1
                        print(f"{name:40}: {module}")

        acc1 = self.evaluate(-1, export_features=self.args.export_features)
        rich.print(f"==> Modal: {self.config.MODAL}, Accuracy: {acc1:.4f}\n")
        
        # write results to self.run_dir
        with open(join(self.run_dir, 'results.txt'), 'a') as f:
            # log time
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{self.launch_command}\n")
            f.write(f"==> Modal: {self.config.MODAL}, Accuracy: {acc1:.4f}\n\n")
