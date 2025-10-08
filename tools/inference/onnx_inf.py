"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import os
import cv2
import glob
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw
from copy import deepcopy
from annotator import Annotator
from annotator_crowdpose import AnnotatorCrowdpose

annotators = {'COCO': Annotator, 'CrowdPose': AnnotatorCrowdpose}

def process_image(sess, im_pil):
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None]

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0)
    annotator = annotators[annotator_type](deepcopy(im_pil))


    output = sess.run(
        output_names=None,
        input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
    )

    scores, labels, keypoints = output
    scores, labels, keypoints = scores[0], labels[0], keypoints[0]
    for kpt, score in zip(keypoints, scores):
        if score > thrh:
            annotator.kpts(
                kpt,
                [h, w]
                )
    annotator.save(f"{OUTPUT_NAME}.jpg")


def process_video(sess, video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{OUTPUT_NAME}.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
        
    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([w, h])[None]
        annotator = annotators[annotator_type](deepcopy(frame_pil))

        im_data = transforms(frame_pil).unsqueeze(0)

        output = sess.run(
            output_names=None,
            input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
        )

        scores, labels, keypoints = output
        scores, labels, keypoints = scores[0], labels[0], keypoints[0]
        for kpt, score in zip(keypoints, scores):
            if score > thrh:
                annotator.kpts(
                    kpt,
                    [h, w]
                    )

        # Convert back to OpenCV image
        frame = annotator.result()

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved as '{OUTPUT_NAME}.mp4'.")

def process_file(sess, file_path):
    # Check if the input file is an image or a video
    try:
        # Try to open the input as an image
        im_pil = Image.open(file_path).convert("RGB")
        process_image(sess, im_pil)
    except IOError:
        # Not an image, process as video
        process_video(sess, file_path)

def main(args):
    assert args.annotator.lower() in ['coco', 'crowdpose']
    # Global variable
    global OUTPUT_NAME, thrh, annotator_type

    """Main function."""
    # Load the ONNX model
    sess = ort.InferenceSession(args.onnx)
    print(f"Using device: {ort.get_device()}")

    input_path = args.input
    thrh = 0.5 if args.thrh is None else args.thrh

    annotator_name = args.annotator.lower()
    if annotator_name == 'coco':
        annotator_type = 'COCO'
    elif annotator_name == 'crowdpose':
        annotator_type = 'CrowdPose'

    # Check if the input argumnet is a file or a folder
    file_path = args.input
    if os.path.isdir(file_path):
        # Process a folder
        folder_dir = args.input
        output_dir = f"{folder_dir}/output"
        os.makedirs(output_dir, exist_ok=True)
        paths = list(glob.iglob(f"{folder_dir}/*.*"))
        for file_path in paths:
            OUTPUT_NAME = file_path.replace(f'{folder_dir}/', f'{output_dir}/').split('.')[0]
            OUTPUT_NAME = f"{OUTPUT_NAME}_{annotator_type}"
            process_file(sess, file_path)
    else:
        # Process a file
        OUTPUT_NAME = f'onxx_results_{annotator_type}'
        process_file(sess, file_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--annotator", type=str, required=True, help="Annotator type: COCO or CrowdPose.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image or video file.")
    parser.add_argument("-t", "--thrh", type=float, required=False, default=None)
    args = parser.parse_args()
    main(args)
