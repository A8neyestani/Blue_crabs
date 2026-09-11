"""Blue crab instance segmentation and experimental projected-size estimation.

Author: Arman Neyestani
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Instance:
    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def segment(model, image, confidence=0.30, iou=0.45, device="cpu"):
    """Return separate instances with binary masks in original image coordinates."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 BGR image.")
    result = model.predict(image, conf=confidence, iou=iou, device=device,
                           retina_masks=True, verbose=False)[0]
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.cpu().numpy() > 0.5
    if masks.shape[1:] != image.shape[:2]:
        raise ValueError("Model masks do not match the original image dimensions.")
    return [Instance(int(box.cls.item()), result.names[int(box.cls.item())],
                     float(box.conf.item()), mask)
            for box, mask in zip(result.boxes, masks)]


def projected_dimensions(mask, depth_m, intrinsics):
    """PCA extents in the camera X/Y plane, in mm; None for insufficient depth.

    These are whole-mask extents, not anatomical carapace measurements or 3D
    surface lengths. Depth must be metric and aligned to the RGB image.
    """
    if mask.ndim != 2 or depth_m.shape != mask.shape:
        raise ValueError("Depth and mask must have the same H x W shape.")
    k = np.asarray(intrinsics, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose([k[0, 1], k[1, 0]], [0, 0])):
        raise ValueError("Expected a finite, zero-skew pinhole camera matrix with positive fx/fy.")
    ys, xs = np.where(mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0))
    if len(xs) < 10:
        return None
    z = depth_m[ys, xs]
    points = np.column_stack(((xs - k[0, 2]) * z / k[0, 0],
                              (ys - k[1, 2]) * z / k[1, 1]))
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    extents = np.ptp(centered @ axes.T, axis=0) * 1000
    return {"pca_axis_1_mm": float(extents[0]), "pca_axis_2_mm": float(extents[1]),
            "valid_depth_pixels": int(len(xs))}


def load_depth(path, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth scale must be a positive number of meters per stored unit.")
    depth = (np.load(path, allow_pickle=False) if Path(path).suffix.lower() == ".npy"
             else cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    if depth is None or depth.ndim != 2:
        raise ValueError("Expected a single-channel depth image or H x W .npy array.")
    if depth.dtype == np.uint8:
        raise ValueError("8-bit depth previews are not accepted as calibrated metric depth.")
    return depth.astype(np.float64) * scale


def overlay(image, instances):
    output = image.copy()
    colors = [(210, 95, 35), (45, 180, 230), (85, 180, 75), (165, 75, 190)]
    for index, item in enumerate(instances):
        color = colors[item.class_id % len(colors)]
        output[item.mask] = (0.55 * output[item.mask] + 0.45 * np.array(color)).astype(np.uint8)
        ys, xs = np.where(item.mask)
        if len(xs):
            cv2.rectangle(output, (int(xs.min()), int(ys.min())),
                          (int(xs.max()), int(ys.max())), color, 2)
            label = f"#{index + 1} {item.class_name.replace('_', ' ')} {item.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = min(int(xs.min()), max(0, image.shape[1] - tw - 8))
            ty = max(th + 8, int(ys.min()) - 5)
            cv2.rectangle(output, (tx, ty - th - 6), (tx + tw + 6, ty + 4), color, -1)
            cv2.putText(output, label, (tx + 3, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Save an overlay, instance masks and JSON.")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    predict.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    predict.add_argument("--confidence", type=float, default=0.30)
    predict.add_argument("--iou", type=float, default=0.45)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--depth", type=Path)
    predict.add_argument("--depth-scale", type=float, help="Meters per stored depth unit; e.g. 0.001 for mm.")
    predict.add_argument("--intrinsics", type=Path, help="JSON file containing a 3 x 3 camera matrix.")
    train = commands.add_parser("train", help="Train with recovered experiment settings.")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--config", type=Path, default=Path(__file__).with_name("train.yaml"))
    evaluate = commands.add_parser("evaluate", help="Evaluate a checkpoint on val or test.")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        if args.command != "train" and not args.weights.is_file():
            raise ValueError(f"Weights not found: {args.weights}. See the download instructions in README.md.")
        from ultralytics import YOLO
        if args.command == "train":
            import yaml
            config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            model = YOLO(config.pop("model"))
            model.train(data=str(args.data.resolve()), device=args.device, project="runs/train", **config)
            return
        model = YOLO(str(args.weights))
        if args.command == "evaluate":
            model.val(data=str(args.data.resolve()), split=args.split, device=args.device,
                      imgsz=640, batch=1, project="runs/evaluate", plots=True)
            return
        if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
            raise ValueError("Confidence and IoU must be between 0 and 1.")
        depth_options = [args.depth is not None, args.depth_scale is not None, args.intrinsics is not None]
        if any(depth_options) and not all(depth_options):
            raise ValueError("Supply --depth, --depth-scale and --intrinsics together.")
        image = read_image(args.image)
        depth = load_depth(args.depth, args.depth_scale) if args.depth else None
        k = np.asarray(json.loads(args.intrinsics.read_text()), dtype=float) if args.intrinsics else None
        if depth is not None:
            projected_dimensions(np.zeros(image.shape[:2], dtype=bool), depth, k)
        instances = segment(model, image, args.confidence, args.iou, args.device)
        args.output.mkdir(parents=True, exist_ok=True)
        records = []
        for i, item in enumerate(instances, start=1):
            filename = f"instance_{i:03d}.png"
            if not cv2.imwrite(str(args.output / filename), item.mask.astype(np.uint8) * 255):
                raise ValueError(f"Cannot write mask: {filename}")
            record = {"instance_id": i, "class_id": item.class_id, "class_name": item.class_name,
                      "confidence": item.confidence, "mask": filename}
            if depth is not None:
                record["projected_dimensions"] = projected_dimensions(item.mask, depth, k)
            records.append(record)
        if not cv2.imwrite(str(args.output / "overlay.jpg"), overlay(image, instances)):
            raise ValueError("Cannot write overlay image.")
        (args.output / "predictions.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(f"Saved {len(records)} instance(s) to {args.output}")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
