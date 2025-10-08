# Custom Pose Estimation Training Guide

## Dataset Structure (COCO Format)
```bash
data/
├── train/
├── val/
├── test/ (optional)
└── annotations/
  ├── train.json
  ├── val.json
  └── test.json (optional)
```

**Notes:**
- If a test set is included, update `dataset_test` in `configs/detrpose/include/dataset.py`.
- If the dataset structure differs from the above, modify `dataset_train`, `dataset_val`, `dataset_test`, and `evaluator` in `configs/detrpose/include/dataset.py`.

## Data Format (COCO)

Annotations must follow the COCO format:

```json
"annotations": [
    {
        "<other_info>": ...,
        "keypoints": [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17],
        "num_keypoints": 15,  // Count of visible keypoints (v > 0)
        "bbox": [x, y, width, height]
    }
]
```

# Training with Custom Poses
1. Ensure the dataset and its configuration align with the expected structure in `configs/detrpose/include/dataset.py`

2. Update num_body_points at `configs/detrpose/include/detrpose_hgnetv2.py`

3. Setup environment
    ```bash
    conda create -n detrpose python=3.11.9
    conda activate detrpose
    pip install -r requirements.txt
    ```

4. Train model
    ```bash
    export model=l
    CUDA_VISIBLE_DEVICES=0 train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --pretrain dfine_${model}_obj365 
    ```