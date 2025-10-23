import json
import os
import cv2
import torch
import numpy as np
from PIL import Image
from albumentations import (
    Compose, RandomCrop, ColorJitter, Resize,
    HorizontalFlip, ShiftScaleRotate, Normalize
)
from albumentations.pytorch import ToTensorV2

# === Config ===
COCO_JSON = r"C:\Users\bmhun\Documents\WORK\dopikAI\DETRPose\data\annotations\train.json"
IMAGE_ROOT = r"C:\Users\bmhun\Documents\WORK\dopikAI\DETRPose\data\train"
SAVE_DIR = "debug_transforms"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load COCO ===
with open(COCO_JSON, "r") as f:
    coco = json.load(f)

# Build image/annotation lookup
ann_by_image = {}
for ann in coco["annotations"]:
    ann_by_image.setdefault(ann["image_id"], []).append(ann)

# === Albumentations transform pipeline ===
transform = Compose([
    # RandomCrop(height=384, width=384, p=1.0),
    HorizontalFlip(p=1.0),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=45, p=1.0),
    ColorJitter(p=1.0),
    Resize(512, 512),
    Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2(),
],
    keypoint_params=dict(format='xy', remove_invisible=False),
    bbox_params=dict(format='coco', label_fields=['category_ids'])  # Add bbox support
)

# === Visualization parameters ===
# Colors for keypoints and bboxes
KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0)
]
BBOX_COLOR = (0, 255, 0)  # Green
KEYPOINT_RADIUS = 3
BBOX_THICKNESS = 2
KEYPOINT_THICKNESS = 2

# === Helper function to draw keypoints and bboxes ===
def draw_keypoints_and_bbox(image, keypoints, bbox=None, visibility=None):
    """
    Draw keypoints and bounding box on image
    
    Args:
        image: numpy array image (H, W, C)
        keypoints: list of [x, y, v] or (x, y) tuples
        bbox: [x, y, width, height] or None
        visibility: list of visibility flags or None
    """
    img_viz = image.copy()
    
    # Draw bounding box
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img_viz, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if len(kp) == 3:
            x, y, v = kp
        else:  # Assume (x, y) format
            x, y = kp
            v = 1  # Assume visible
        
        x, y = int(x), int(y)
        
        # Only draw visible keypoints (v > 0)
        if v > 0:
            color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
            cv2.circle(img_viz, (x, y), KEYPOINT_RADIUS, color, -1)
            cv2.putText(img_viz, str(i), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return img_viz

# === Test loop ===
results = []

for img_info in coco["images"]:
    img_path = os.path.join(IMAGE_ROOT, img_info["file_name"])
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Cannot read {img_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get first annotation for this image
    anns = ann_by_image.get(img_info["id"], [])
    if not anns:
        continue

    ann = anns[0]
    keypoints = np.array(ann.get("keypoints", [])).reshape(-1, 3)
    bbox = ann.get("bbox", None)  # [x, y, width, height]

    # Separate (x, y, v)
    kpts_xy = [(x, y) for x, y, v in keypoints]
    visibility = [v for x, y, v in keypoints]
    category_ids = [ann.get("category_id", 1)]  # For bbox_params

    try:
        # Apply transform with both keypoints and bbox
        if bbox:
            transformed = transform(
                image=image, 
                keypoints=kpts_xy, 
                bboxes=[bbox],
                category_ids=category_ids
            )
        else:
            transformed = transform(
                image=image, 
                keypoints=kpts_xy
            )
    except Exception as e:
        print(f"❌ Failed transform on {img_info['file_name']}: {e}")
        continue

    img_tensor = transformed["image"]
    kpts_trans = transformed["keypoints"]
    bboxes_trans = transformed.get("bboxes", [])

    # Reattach visibility to transformed keypoints
    keypoints_final = [[x, y, v] for (x, y), v in zip(kpts_trans, visibility)]

    # Convert tensor → numpy for saving
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = (img_np * 255).astype('uint8')

    # === Create visualizations ===
    
    # 1. Original image with keypoints and bbox
    original_viz = draw_keypoints_and_bbox(
        image, 
        keypoints, 
        bbox=bbox,
        visibility=visibility
    )
    
    # 2. Transformed image with transformed keypoints and bbox
    transformed_bbox = bboxes_trans[0] if bboxes_trans else None
    transformed_viz = draw_keypoints_and_bbox(
        img_np, 
        keypoints_final, 
        bbox=transformed_bbox,
        visibility=visibility
    )
    
    # 3. Side-by-side comparison
    h1, w1 = original_viz.shape[:2]
    h2, w2 = transformed_viz.shape[:2]
    
    # Resize to same height for side-by-side
    if h1 != h2:
        scale = h1 / h2
        new_w2 = int(w2 * scale)
        transformed_viz = cv2.resize(transformed_viz, (new_w2, h1))
        h2, w2 = transformed_viz.shape[:2]
    
    side_by_side = np.hstack([original_viz, transformed_viz])
    
    # Add labels to side-by-side
    cv2.putText(side_by_side, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(side_by_side, "Transformed", (w1 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save all visualizations
    base_name = img_info["file_name"].replace('.jpg', '')
    
    # Save original with annotations
    orig_save_path = os.path.join(SAVE_DIR, f"{base_name}_original.jpg")
    cv2.imwrite(orig_save_path, cv2.cvtColor(original_viz, cv2.COLOR_RGB2BGR))
    
    # Save transformed with annotations
    trans_save_path = os.path.join(SAVE_DIR, f"{base_name}_transformed.jpg")
    cv2.imwrite(trans_save_path, cv2.cvtColor(transformed_viz, cv2.COLOR_RGB2BGR))
    
    # Save side-by-side comparison
    comparison_save_path = os.path.join(SAVE_DIR, f"{base_name}_comparison.jpg")
    cv2.imwrite(comparison_save_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Saved visualizations for: {base_name}")
    print(f"   - Original: {orig_save_path}")
    print(f"   - Transformed: {trans_save_path}")
    print(f"   - Comparison: {comparison_save_path}")

    results.append((img_tensor, keypoints_final))

print(f"\n✅ Finished testing Albumentations transforms ({len(results)} images processed).")
print(f"Visualizations saved in: {SAVE_DIR}")

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
      bbox_params=dict(format='coco', label_fields=['category_ids'])  # Add bbox support
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
  )