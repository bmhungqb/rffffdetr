# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
# import datasets.transforms as T

__all__ = ['build']


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        super(CocoDetection, self).__init__()
        self.img_folder = Path(img_folder)
        self.coco = COCO(ann_file)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._transforms = transforms
        
        imgIds = sorted(self.coco.getImgIds())
        self.all_imgIds = []
        
        if "train" in ann_file:
            for image_id in imgIds:
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                if not ann_ids:
                    continue
                target = self.coco.loadAnns(ann_ids)
                num_keypoints = [obj["num_keypoints"] for obj in target]
                if sum(num_keypoints) == 0:
                    continue
                self.all_imgIds.append(image_id)
        else:
            for image_id in imgIds:
                self.all_imgIds.append(image_id)

    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __len__(self):
        return len(self.all_imgIds)

    def load_item(self, idx):
        image_id = self.all_imgIds[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target = self.coco.loadAnns(ann_ids)

        target = {'image_id': image_id, 'annotations': target}
        img = Image.open(self.img_folder / self.coco.loadImgs(image_id)[0]['file_name'])
        img, target = self.prepare(img, target)
        return img, target

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img_np = np.array(img)

            transformed = self._transforms(
                image=img_np,
                bboxes=target.get("boxes", []),
                labels=target.get("labels", []),
                keypoints=target.get("keypoints", []),
            )

            # extract transformed
            img = transformed["image"]
            target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
            if "keypoints" in transformed:
                target["keypoints"] = torch.as_tensor(transformed["keypoints"], dtype=torch.float32)
        
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        # filter anno
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        anno = [obj for obj in anno if obj['num_keypoints'] != 0]
        # FIX: get num_keyspoints
        num_keypoints = len(target["categories"]["keypoints"]) if "categories" in target and "keypoints" in target["categories"] else max(len(obj["keypoints"]) for obj in target["annotations"]) // 3
        keypoints = [obj["keypoints"] for obj in anno]
        boxes = [obj["bbox"] for obj in anno]
        # Handle empty annotations
        if len(anno) == 0:
            empty_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "keypoints": torch.zeros((0, 0, 3), dtype=torch.float32),
                "image_id": image_id,
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "orig_size": torch.as_tensor([int(w), int(h)]),
                "size": torch.as_tensor([int(h), int(w)]),
            }
            if self.return_masks:
                empty_target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            return image, empty_target
        
        # === Extract fields ===
        num_keypoints = len(anno[0]["keypoints"]) // 3
        boxes = torch.as_tensor([obj["bbox"] for obj in anno], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w - 1)
        boxes[:, 1::2].clamp_(min=0, max=h - 1)

        classes = torch.as_tensor([obj["category_id"] for obj in anno], dtype=torch.int64)
        keypoints = torch.as_tensor([obj["keypoints"] for obj in anno], dtype=torch.float32).reshape(-1, num_keypoints, 3)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        else:
            masks = None

        # === Filter invalid boxes ===
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes, classes, keypoints = boxes[keep], classes[keep], keypoints[keep]
        if masks is not None:
            masks = masks[keep]

        # === Prepare final target ===
        target = {
            "boxes": boxes,
            "labels": classes,
            "keypoints": keypoints,
            "image_id": image_id,
            "area": torch.as_tensor([obj["area"] for obj in anno], dtype=torch.float32)[keep],
            "iscrowd": torch.as_tensor([obj.get("iscrowd", 0) for obj in anno], dtype=torch.int64)[keep],
            "orig_size": torch.as_tensor([int(w), int(h)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }
        if masks is not None:
            target["masks"] = masks

        return image, target


