# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
from torch.utils.data import DataLoader

from util.slconfig import SLConfig
import util.misc as utils

import datasets
from datasets import build_dataset, BatchImageCollateFunction


def create(args, classname):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    class_module = getattr(args, classname)
    assert class_module in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(class_module)
    return build_func(args)

def main(args):
    cfg = SLConfig.fromfile(args.config)
    device = args.device

    setattr(cfg, 'coco_path', args.data_path)
    setattr(cfg, 'batch_size_train', 1)
    setattr(cfg, 'batch_size_val', 1)

    if 'HGNetv2' in cfg.backbone:
        cfg.pretrained = False

    # build model
    model, _ = create(cfg, 'modelname')
    model.to(device)

    dataset_val = build_dataset(image_set='val', args=cfg)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=BatchImageCollateFunction(), num_workers=4)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        model.load_state_dict(state)

    # folder path
    main_folder = cfg.output_dir
    if 'data/wireframe_processed' in args.data_path:
        backbone_dir = f'{main_folder}/visualization/backbone_wireframe'
        encoder_dir = f'{main_folder}/visualization/encoder_wireframe'

    elif 'data/york_processed' in args.data_path:
        backbone_dir = f'{main_folder}/visualization/backbone_york'
        encoder_dir = f'{main_folder}/visualization/encoder_york'
    else:
        raise 'Dataset does not exist. We support only wireframe and york datasets'

    os.makedirs(backbone_dir , exist_ok=True)
    os.makedirs(encoder_dir, exist_ok=True)

    with torch.no_grad():

        for i, (samples, targets) in enumerate(data_loader_val):
            samples = samples.to(device)

            enc_feature_maps = []
            backbone_feature_maps = []
            hooks = [
                model.backbone.register_forward_hook(
                    lambda self, input, output: backbone_feature_maps.append(output)
                ),
                model.encoder.register_forward_hook(
                    lambda self, input, output: enc_feature_maps.append(output)
                ),
            ]
            model(samples)
            
            for hook in hooks:
                hook.remove()    
 
            back_feats = backbone_feature_maps[0]    
            enc_feats = enc_feature_maps[0]

            curr_img_id = targets[0]['image_id'].tolist()[0]

            for j, back_feat in enumerate(back_feats):
                down = j + 1

                back_feat = back_feat[0].mean(0).cpu()

                fig = plt.figure(figsize=(16, 16))
                plt.axis('off')
                plt.imshow(back_feat)
                plt.savefig(
                    f"{backbone_dir}/{curr_img_id}_ds_{down}.png", 
                    bbox_inches='tight', 
                    pad_inches=0, 
                    dpi=200
                    )
                plt.close()

            for j, enc_feat in enumerate(enc_feats):
                down = j + 1

                enc_feat = enc_feat[0].mean(0).cpu()

                fig = plt.figure(figsize=(16, 16))
                plt.axis('off')
                plt.imshow(enc_feat)
                plt.savefig(
                    f"{encoder_dir}/{curr_img_id}_ds_{down}.png", 
                    bbox_inches='tight', 
                    pad_inches=0, 
                    dpi=200
                    )
                plt.close()

            # check condition to stop program
            if args.num_images is not None and i + 1 >= args.num_images:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization of Deformable Line Attention')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', default='', help='resume from checkpoint')
    parser.add_argument('-p', '--data-path', type=str, default='data/wireframe_processed', help='data path')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-n', '--num_images', type=int, help='total number of images to plot')
    args = parser.parse_args()
    main(args)
