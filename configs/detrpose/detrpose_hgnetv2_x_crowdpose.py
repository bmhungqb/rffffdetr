from .include.detrpose_hgnetv2 import model, criterion, training_params, postprocessor
from .include.dataset_crowdpose import dataset_train, dataset_val, dataset_test, evaluator

from src.core import LazyCall as L
from src.nn.optimizer import ModelEMA 
from src.misc.get_param_dicts import get_optim_params

from torch import optim

training_params.output_dir =  "output/detrpose_hgnetv2_x_crowdpose"
training_params.epochs = 52 # 48 + 4
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

model.backbone.name = 'B5'
model.encoder.hidden_dim = 384
model.encoder.dim_feedforward = 2048
model.transformer.hidden_dim = 384
# model.transformer.feat_channels = [384, 384, 384]
model.transformer.reg_scale = 8
