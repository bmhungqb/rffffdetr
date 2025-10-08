import argparse
from omegaconf import OmegaConf

from src.solver import Trainer
from src.misc import dist_utils
from src.core import LazyConfig, instantiate

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--pretrain', default=None, help='apply transfer learning to the backbone and encoder using DFINE weights')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    return parser

def main(args):
    cfg = LazyConfig.load(args.config_file)

    updates = OmegaConf.create()
    for k, v in args.__dict__.items():
        if k not in ["options"] and v is not None:
            updates[k] = v
    cfg.training_params = OmegaConf.merge(cfg.training_params, updates)

    if args.options:
        cfg = LazyConfig.apply_overrides(cfg, args.options) 
    print(cfg)
    
    solver = Trainer(cfg)

    assert not(args.eval and args.test), "you can't do evaluation and test at the same time"

    if args.eval:
        if hasattr(cfg.model.backbone, 'pretrained'):
            cfg.model.backbone.pretrained = False
        solver.eval()
    elif args.test:
        if hasattr(cfg.model.backbone, 'pretrained'):
            cfg.model.backbone.pretrained = False
        solver.test()
    else:
        solver.fit()
    dist_utils.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RT-GroupPose training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
