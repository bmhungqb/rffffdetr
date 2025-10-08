#########################################################################################
# Modified from:
#   Ultralytics 
#   https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/plotting.py
#########################################################################################

import math
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from annotator import Annotator, Colors


colors = Colors()  # create instance for 'from utils.plots import colors'

class AnnotatorCrowdpose(Annotator):
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image | np.ndarray): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype | ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
        dark_colors (set): Set of colors considered dark for text contrast.
        light_colors (set): Set of colors considered light for text contrast.

    Examples:
        >>> from ultralytics.utils.plotting import Annotator
        >>> im0 = cv2.imread("test.png")
        >>> annotator = Annotator(im0, line_width=10)
        >>> annotator.box_label([10, 10, 100, 100], "person", (255, 0, 0))
    """

    def __init__(
        self,
        im,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        super().__init__(im, line_width, font_size, font, pil, example)

        # Pose Crowdpose
        self.skeleton = [
            # limbs
            [12, 10],
            [10, 8],
            [11, 9],
            [9, 7],
            # torso
            [8, 7],
            [8, 2],
            [7, 1],
            # arms
            [14, 1],
            [14, 2],
            [1, 3],
            [3, 5],
            [2, 4],
            [4, 6],
            # head
            [14, 13],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 0, 16]]
        self.kpt_color = colors.pose_palette[[0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 16, 0]]
        # 9, 9, 9, 9, 9, 9, 9, 0, 16, 16, 0, 0, 0, 0, 0, 0]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    # def kpts(
    #     self,
    #     kpts,
    #     shape: tuple = (640, 640),
    #     radius: Optional[int] = None,
    #     kpt_line: bool = True,
    #     conf_thres: float = 0.25,
    #     kpt_color: Optional[tuple] = None,
    # ):
    #     """
    #     Plot keypoints on the image.

    #     Args:
    #         kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
    #         shape (tuple, optional): Image shape (h, w).
    #         radius (int, optional): Keypoint radius.
    #         kpt_line (bool, optional): Draw lines between keypoints.
    #         conf_thres (float, optional): Confidence threshold.
    #         kpt_color (tuple, optional): Keypoint color (B, G, R).

    #     Note:
    #         - `kpt_line=True` currently only supports human pose plotting.
    #         - Modifies self.im in-place.
    #         - If self.pil is True, converts image to numpy array and back to PIL.
    #     """
    #     radius = radius if radius is not None else self.lw
    #     if self.pil:
    #         # Convert to numpy first
    #         self.im = np.asarray(self.im).copy()
    #     nkpt, ndim = kpts.shape
    #     is_pose = nkpt == 17 and ndim in {2, 3}
    #     kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
    #     for i, k in enumerate(kpts):
    #         color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
    #         x_coord, y_coord = k[0], k[1]
    #         if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
    #             if len(k) == 3:
    #                 conf = k[2]
    #                 if conf < conf_thres:
    #                     continue
    #             cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    #     if kpt_line:
    #         ndim = kpts.shape[-1]
    #         for i, sk in enumerate(self.skeleton):
    #             pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
    #             pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
    #             if ndim == 3:
    #                 conf1 = kpts[(sk[0] - 1), 2]
    #                 conf2 = kpts[(sk[1] - 1), 2]
    #                 if conf1 < conf_thres or conf2 < conf_thres:
    #                     continue
    #             if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
    #                 continue
    #             if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
    #                 continue
    #             cv2.line(
    #                 self.im,
    #                 pos1,
    #                 pos2,
    #                 kpt_color or self.limb_color[i].tolist(),
    #                 thickness=int(np.ceil(self.lw / 2)),
    #                 lineType=cv2.LINE_AA,
    #             )
    #     if self.pil:
    #         # Convert im back to PIL and update draw
    #         self.fromarray(self.im)
