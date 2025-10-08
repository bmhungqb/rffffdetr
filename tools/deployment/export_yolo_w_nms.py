import os
from ultralytics import YOLO

def main(args):
    output_folder = 'trt_engines'
    os.makedirs(output_folder, exist_ok=True)

    model = YOLO(f"{args.name}.pt")
    model.export(format="engine", nms=True, iou=args.iou_threshold, conf=args.score_threshold, half=True, dynamic=False)

    with open(f"{args.name}.engine", "rb") as f:
        meta_len = int.from_bytes(f.read(4), byteorder="little")
        f.seek(meta_len + 4)
        engine = f.read()

    new_name  = f"{args.name}_" + str(args.iou_threshold).split('.')[1] + '_' + str(args.score_threshold).split('.')[1]
    with open(f"{output_folder}/{new_name}.engine", "wb") as f:
        f.write(engine)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="yolo11n_tuned")
    parser.add_argument("--score_threshold", type=float, default=0.01)
    parser.add_argument("--iou_threshold", type=float, default=0.7)
    args = parser.parse_args()

    main(args)