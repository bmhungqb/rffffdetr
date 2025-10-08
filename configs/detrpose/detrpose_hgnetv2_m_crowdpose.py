from .include.detrpose_hgnetv2 import model, criterion, training_params, postprocessor
from .include.dataset_crowdpose import dataset_train, dataset_val, dataset_test, evaluator

from src.core import LazyCall as L
from src.nn.optimizer import ModelEMA 
from src.misc.get_param_dicts import get_optim_params

from torch import optim

training_params.output_dir =  "output/detrpose_hgnetv2_m_crowdpose"
training_params.epochs = 76 # 72 + 4
training_params.use_ema = True

ema = L(ModelEMA)(
    decay=0.9999,
    warmups=2000
    )

# optimizer params
optimizer = L(optim.AdamW)(
    params=L(get_optim_params)(
        cfg=[
                {
                'params': '^(?=.*backbone).*$',
                'lr': 0.00001
                },
            ],
        # model=model
        ),
    lr=0.0001,
    betas=[0.9, 0.999],
    weight_decay=0.0001
    )

lr_scheduler = L(optim.lr_scheduler.MultiStepLR)(
    # optimizer=optimizer,
    milestones=[1000],
    gamma=0.1
    )

model.transformer.num_body_points=14
criterion.matcher.num_body_points=14
criterion.num_body_points=14
postprocessor.num_body_points=14

model.backbone.name = 'B2'
model.backbone.use_lab = True
model.encoder.in_channels = [384, 768, 1536]
model.encoder.depth_mult = 0.67
model.transformer.num_decoder_layers = 4

dataset_train.dataset.transforms.policy = {
    'name': 'stop_epoch',
    'ops': ['Mosaic', 'RandomCrop', 'RandomZoomOut'],
    'epoch': [5, 41, 72] # 72 / 2 + 5 = 35
    }
dataset_train.collate_fn.base_size_repeat = 6
dataset_train.collate_fn.stop_epoch = 72
