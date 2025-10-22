from src.core import LazyCall as L
from src.data import CocoDetection
from src.data.dataloader import (
	BatchImageCollateFunction, 
	DataLoader
	)
from src.data.coco_eval import CocoEvaluator
from src.data.container import Compose
import src.data.transforms as T
from albumentations import (
    Compose, RandomCrop, ColorJitter, Resize,
    HorizontalFlip, ShiftScaleRotate, Normalize
)
from albumentations.pytorch import ToTensorV2
from .detrpose_hgnetv2 import eval_spatial_size

from omegaconf import OmegaConf

scales = [(640, 640)]
max_size = 1333
scales2_resize = [400, 500, 600]

__all__ = ["dataset_train", "dataset_val", "dataset_test", "evaluator"]

dataset_train = L(DataLoader)(
	dataset=L(CocoDetection)(
		img_folder="data/train",
		ann_file="data/annotations/train.json",
    transforms=Compose([
        # RandomCrop(height=640, width=640, p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=30, p=0.5),
        ColorJitter(p=0.5),
        Resize(height=scales[0][0], width=scales[0][0]),
        Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ],
      keypoint_params=dict(format='xy', remove_invisible=False),
      bbox_params=dict(format='coco', label_fields=['labels'])
		)
  ),
	total_batch_size=16,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		base_size_repeat=4,
		stop_epoch=48,
		),
	num_workers=4,
	shuffle=True,
	drop_last=True,
	pin_memory=True
  )

dataset_val = L(DataLoader)(
	dataset=L(CocoDetection)(
		img_folder="data/val",
		ann_file="data/annotations/val.json",
		transforms=Compose([
      Resize(height=eval_spatial_size[0], width=eval_spatial_size[1]),
      Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
      ToTensorV2()
    ])
		),
	total_batch_size=32,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		),
	num_workers=4,
	shuffle=False,
	drop_last=False,
	pin_memory=True
	)

dataset_test = L(DataLoader)(
	dataset=L(CocoDetection)(
		img_folder="data/val",
		ann_file="data/annotations/val.json",
		transforms=Compose([
        Resize(height=eval_spatial_size[0], width=eval_spatial_size[1]),
        Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2()]
			),
		),
	total_batch_size=32,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		),
	num_workers=4,
	shuffle=False,
	drop_last=False,
	pin_memory=True
	)

evaluator = L(CocoEvaluator)(
	ann_file="data/annotations/val.json",
	iou_types=['keypoints'],
	useCats=True
	)