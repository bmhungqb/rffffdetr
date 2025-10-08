# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random, os
import PIL
import torch
import numbers
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as F2

import math
from PIL import Image
from ..misc.mask_ops import interpolate
from ..misc.box_ops import box_xyxy_to_cxcywh

from omegaconf import ListConfig

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "keypoints"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "keypoints" in target:
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        keypoints = target["keypoints"]
        cropped_keypoints = keypoints[...,:2] - torch.as_tensor([j, i])[None, None]
        cropped_viz = keypoints[..., 2:]

        # keep keypoint if 0<=x<=w and 0<=y<=h else remove
        cropped_viz = torch.where(
            torch.logical_and( # condition to know if keypoint is inside the image
                torch.logical_and(0<=cropped_keypoints[..., 0].unsqueeze(-1), cropped_keypoints[..., 0].unsqueeze(-1)<=w), 
                torch.logical_and(0<=cropped_keypoints[..., 1].unsqueeze(-1), cropped_keypoints[..., 1].unsqueeze(-1)<=h)
                ),
            cropped_viz, # value if condition is True
            0 # value if condition is False
            )

        cropped_keypoints = torch.cat([cropped_keypoints, cropped_viz], dim=-1)
        cropped_keypoints = torch.where(cropped_keypoints[..., -1:]!=0, cropped_keypoints, 0)

        target["keypoints"] = cropped_keypoints

        keep = cropped_viz.sum(dim=(1, 2)) != 0

    # remove elements for which the no keypoint is on the image
    for field in fields:
        target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "keypoints" in target:
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        keypoints = target["keypoints"]
        # keypoints[:,:,0] = w - keypoints[:,:, 0]
        keypoints[:,:,0] = torch.where(keypoints[..., -1]!=0, w - keypoints[:,:, 0]-1, 0)
        for pair in flip_pairs:
            keypoints[:,pair[0], :], keypoints[:,pair[1], :] = keypoints[:,pair[1], :], keypoints[:,pair[0], :].clone()
        target["keypoints"] = keypoints

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple, ListConfig)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height, 1])
        target["keypoints"] = scaled_keypoints

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, padding)
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], padding)

    if "keypoints" in target:
        keypoints = target["keypoints"]
        padped_keypoints = keypoints.view(-1, 3)[:,:2] + torch.as_tensor(padding[:2])
        padped_keypoints = torch.cat([padped_keypoints, keypoints.view(-1, 3)[:,2].unsqueeze(1)], dim=1)
        padped_keypoints = torch.where(padped_keypoints[..., -1:]!=0, padped_keypoints, 0)
        target["keypoints"] = padped_keypoints.view(target["keypoints"].shape[0], -1, 3)

    if "boxes" in target:
        boxes = target["boxes"]
        padded_boxes = boxes + torch.as_tensor(padding)
        target["boxes"] = padded_boxes


    return padded_image, target


class RandomZoomOut(object):
    def __init__(self, p=0.5, side_range=[1, 2.5]):
        self.p = p
        self.side_range = side_range

    def __call__(self, img, target):
        if random.random() < self.p:
            ratio = float(np.random.uniform(self.side_range[0], self.side_range[1], 1))
            h, w = target['size']
            pad_w = int((ratio-1) * w)
            pad_h = int((ratio-1) * h)
            padding = [pad_w, pad_h, pad_w, pad_h]
            img, target = pad(img, target, padding)
        return img, target


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            region = self.get_params(target)
            return crop(img, target, region)
        return img, target

    @staticmethod
    def get_params(target):
        target = target.copy()
        boxes = target['boxes']
        cases = list(range(len(boxes)))
        idx = random.sample(cases, 1)[0] # xyxy
        box = boxes[idx].clone()
        box[2:] -= box[:2] # top-left-height-width
        # box[2:] *= 1.2 
        box = box[[1, 0, 3, 2]]
        return box.tolist()


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple, ListConfig))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        if "area" in target:
            area = target["area"]
            area = area / (torch.tensor(w, dtype=torch.float32)*torch.tensor(h, dtype=torch.float32))
            target["area"] = area
        else:
            target["area"] = boxes[:, 2] * boxes[:, 3] * 0.53

        if "keypoints" in target:
            keypoints = target["keypoints"]  # (4, 17, 3) (num_person, num_keypoints, 3)
            keypoints = torch.where(keypoints[..., -1:]!=0, keypoints, 0)
            num_body_points = keypoints.size(1)
            V = keypoints[:, :, 2]  # visibility of the keypoints torch.Size([number of persons, 17])
            V[V == 2] = 1
            Z=keypoints[:, :, :2]
            Z = Z.contiguous().view(-1, 2 * num_body_points)
            Z = Z / torch.tensor([w, h] * num_body_points, dtype=torch.float32)
            all_keypoints = torch.cat([Z, V], dim=1)  # torch.Size([number of persons, 2+34+17])
            target["keypoints"] = all_keypoints
        return image, target


class Mosaic(object):
    def __init__(self, output_size=320, max_size=None, probability=1.0, 
        use_cache=False, max_cached_images=50, random_pop=True) -> None:
        super().__init__()
        self.resize = RandomResize(sizes=[output_size], max_size=max_size)
        self.probability = probability

        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F2.get_size if hasattr(F2, "get_size") else F2.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            image, target = dataset.load_item(idx)
            image, target = self.resize(image, target)
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        """Merges targets into a single target dictionary for the mosaic."""
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        offsets_pose = torch.tensor([[0, 0, 0], [max_width, 0, 0], [0, max_height, 0], [max_width, max_height, 0]])
        merged_target = {}
        for key in targets[0]:
            if key == 'boxes':
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            elif key == 'keypoints':
                values = [torch.where(target[key][..., -1:]!=0, target[key] + offsets_pose[i], 0) for i, target in enumerate(targets)]
            else:
                values = [target[key] for target in targets]

            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return merged_image, merged_target

    def __call__(self, image, target, dataset):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        return mosaic_image, mosaic_target

class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.p = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img, target):

        if random.random() < self.p:
            fn_idx = torch.randperm(4)
            for fn_id in fn_idx:
                if fn_id == 0 and self.brightness is not None:
                    brightness = self.brightness
                    brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                    img = F.adjust_brightness(img, brightness_factor)

                if fn_id == 1 and self.contrast is not None:
                    contrast = self.contrast
                    contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                    img = F.adjust_contrast(img, contrast_factor)

                if fn_id == 2 and self.saturation is not None:
                    saturation = self.saturation
                    saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                    img = F.adjust_saturation(img, saturation_factor)

                if fn_id == 3 and self.hue is not None:
                    hue = self.hue
                    hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                    img = F.adjust_hue(img, hue_factor)

        return img, target
    
class Rotate(object):
    def __init__(self, degrees, p=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be non-negative")
            self.degrees = [-degrees, degrees]
        elif isinstance(degrees, (tuple, list)) and len(degrees) == 2:
            if not (-360 <= degrees[0] <= degrees[1] <= 360):
                raise ValueError("Degrees should be between -360 and 360.")
            self.degrees = degrees
        else:
            raise TypeError("Degrees should be a single number or a list/tuple with length 2.")
        self.p = p

    def __call__(self, image, target=None):
        if random.random() >= self.p:
            return image, target

        angle = float(torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item())
        w, h = image.size  # original width, height (PIL coordinates: x right, y down)

        # rotate image (PIL/torchvision rotates counter-clockwise by `angle`)
        rotated_image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR, expand=True)
        new_w, new_h = rotated_image.size

        if target is None:
            return rotated_image, None

        target = target.copy()

        # Use float32 for geometric ops
        device = None
        # --- Prepare rotation coefficients for IMAGE coordinate system (y down) ---
        # For image coords, rotation by angle (counter-clockwise on image) transforms:
        # x' = x * cos(angle) + y * sin(angle)
        # y' = -x * sin(angle) + y * cos(angle)
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # center points
        cx, cy = w / 2.0, h / 2.0
        center_orig = torch.tensor([cx, cy], dtype=torch.float32)

        # --- BOXES (COCO format: x,y,w,h) ---
        if "boxes" in target and target["boxes"] is not None and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone().to(torch.float32)
            device = boxes.device
            # ensure tensors on same device
            boxes = boxes.to(device) if device is not None else boxes
            center = torch.tensor([cx, cy], dtype=boxes.dtype, device=device)

            x_min = boxes[:, 0]
            y_min = boxes[:, 1]
            x_max = x_min + boxes[:, 2]
            y_max = y_min + boxes[:, 3]

            # corners: (N, 4, 2)
            corners = torch.stack([
                torch.stack([x_min, y_min], dim=-1),
                torch.stack([x_max, y_min], dim=-1),
                torch.stack([x_max, y_max], dim=-1),
                torch.stack([x_min, y_max], dim=-1),
            ], dim=1).to(device=device)

            # translate to origin (center)
            corners = corners - center.to(device)

            x = corners[..., 0]
            y = corners[..., 1]

            # rotate in IMAGE coordinates (note +sin in x, -sin in y)
            x_new = x * cos_a + y * sin_a
            y_new = -x * sin_a + y * cos_a
            rotated_corners = torch.stack([x_new, y_new], dim=-1)

            # translate back to new image center
            new_center = torch.tensor([new_w / 2.0, new_h / 2.0], dtype=rotated_corners.dtype, device=device)
            rotated_corners = rotated_corners + new_center

            # mins & maxs
            x_min_new = rotated_corners[..., 0].min(dim=1).values
            y_min_new = rotated_corners[..., 1].min(dim=1).values
            x_max_new = rotated_corners[..., 0].max(dim=1).values
            y_max_new = rotated_corners[..., 1].max(dim=1).values

            new_boxes = torch.stack([
                x_min_new,
                y_min_new,
                x_max_new - x_min_new,
                y_max_new - y_min_new
            ], dim=1)

            # clip per-dimension (use scalar max -> safe)
            new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], min=0.0, max=float(new_w))
            new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], min=0.0, max=float(new_h))
            # recompute x_max/y_max after x_min/y_min clamping then clamp them to bounds
            x_max_clamped = torch.clamp(new_boxes[:, 0] + new_boxes[:, 2], min=0.0, max=float(new_w))
            y_max_clamped = torch.clamp(new_boxes[:, 1] + new_boxes[:, 3], min=0.0, max=float(new_h))
            new_boxes[:, 2] = x_max_clamped - new_boxes[:, 0]
            new_boxes[:, 3] = y_max_clamped - new_boxes[:, 1]

            target["boxes"] = new_boxes
            target["area"] = new_boxes[:, 2] * new_boxes[:, 3]

        # --- KEYPOINTS (N, K, 3) with x,y,visibility ---
        if "keypoints" in target and target["keypoints"] is not None and len(target["keypoints"]) > 0:
            kps = target["keypoints"].clone().to(torch.float32)
            device = kps.device if device is None else device
            kps = kps.to(device) if device is not None else kps

            kp_xy = kps[:, :, :2] - torch.tensor([cx, cy], dtype=torch.float32, device=device)
            x = kp_xy[:, :, 0]
            y = kp_xy[:, :, 1]

            # rotate (image coords)
            x_new = x * cos_a + y * sin_a
            y_new = -x * sin_a + y * cos_a
            rotated_kp_xy = torch.stack([x_new, y_new], dim=-1) + torch.tensor([new_w / 2.0, new_h / 2.0], dtype=torch.float32, device=device)

            # clamp coordinates separately
            rotated_kp_xy[..., 0] = torch.clamp(rotated_kp_xy[..., 0], min=0.0, max=float(new_w))
            rotated_kp_xy[..., 1] = torch.clamp(rotated_kp_xy[..., 1], min=0.0, max=float(new_h))

            visibility = kps[:, :, 2:3]
            # keep visibility values, if visibility==0 set point to 0
            rotated_kp = torch.cat([rotated_kp_xy, visibility], dim=-1)
            rotated_kp = torch.where(visibility != 0, rotated_kp, torch.tensor(0.0, dtype=rotated_kp.dtype, device=device))

            target["keypoints"] = rotated_kp

            # remove instances with no visible keypoints
            keep = (rotated_kp[:, :, 2] != 0).sum(dim=1) != 0
            if keep.numel() > 0:
                for field in ["labels", "area", "iscrowd", "boxes", "masks", "keypoints"]:
                    if field in target and target[field] is not None:
                        try:
                            target[field] = target[field][keep]
                        except Exception:
                            # if indexing fails (e.g., masks in other format), skip
                            pass

        # --- MASKS ---
        if "masks" in target and target["masks"] is not None:
            # rotate mask same as image; keep boolean mask
            try:
                target["masks"] = F.rotate(
                    target["masks"].float(), angle, interpolation=F.InterpolationMode.NEAREST, expand=True
                ) > 0.5
            except Exception:
                # in case masks type isn't a PIL-like tensor, skip or handle externally
                pass

        target["size"] = torch.tensor([new_h, new_w], dtype=torch.int32)

        return rotated_image, target