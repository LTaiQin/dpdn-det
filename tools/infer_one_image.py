import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

# Ensure `second_14.1` repo root is importable when running as:
#   python tools/infer_one_image.py ...
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ovd  # noqa: F401  (register datasets / meta-arch)
from ovd.config import add_ovd_config


def collect_open_vocab_classes(cfg):
    class_names = []
    seen = set()
    for dataset_name in cfg.DATASETS.TRAIN:
        try:
            meta = MetadataCatalog.get(dataset_name)
        except KeyError:
            continue
        candidates = getattr(meta, "open_vocab_classes", getattr(meta, "thing_classes", []))
        for name in candidates:
            normalized = name.strip()
            key = normalized.lower()
            if normalized == "" or key in seen:
                continue
            seen.add(key)
            class_names.append(normalized)
    return class_names


def setup_cfg(args):
    cfg = get_cfg()
    add_ovd_config(cfg)
    cfg.merge_from_file(args.config_file)

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(args.score_thresh)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(args.score_thresh)

    open_vocab_classes = collect_open_vocab_classes(cfg)
    if len(open_vocab_classes) == 0:
        cfg.MODEL.OPEN_VOCAB_CLASS_NAMES = []
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 228
        cfg.MODEL.RETINANET.NUM_CLASSES = 228
    else:
        cfg.MODEL.OPEN_VOCAB_CLASS_NAMES = open_vocab_classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(open_vocab_classes)
        cfg.MODEL.RETINANET.NUM_CLASSES = len(open_vocab_classes)

    cfg.freeze()
    return cfg, open_vocab_classes


def infer_one_image(cfg, image_path: str):
    predictor = DefaultPredictor(cfg)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    with torch.no_grad():
        outputs = predictor(img)
    return img, outputs


def draw_and_save(img_bgr, outputs, class_names, out_path: str):
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    metadata = MetadataCatalog.get("open_vocab_infer")
    if class_names:
        metadata.set(thing_classes=list(class_names))

    viz = Visualizer(
        img_bgr[:, :, ::-1],
        metadata=metadata,
        instance_mode=ColorMode.IMAGE,
    )
    inst = outputs["instances"].to("cpu") if "instances" in outputs else None
    vis_img = viz.draw_instance_predictions(inst).get_image()[:, :, ::-1] if inst is not None else img_bgr
    ok = cv2.imwrite(out_path, vis_img)
    if not ok:
        raise OSError(f"Failed to write output image: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Infer a single image and save visualization.")
    p.add_argument(
        "--config-file",
        default="configs/coco/COCO_OVD_Food2K_PIS.yaml",
        help="Path to config yaml (relative to second_14.1/ or absolute).",
    )
    p.add_argument(
        "--weights",
        default="output/coco_ovd_food2k_PIS_with_cap_new/model_0079999.pth",
        help="Path to model weights (relative to second_14.1/ or absolute).",
    )
    p.add_argument("--input", required=True, help="Input image path.")
    p.add_argument(
        "--output",
        default="output/infer_one_image/vis.jpg",
        help="Output image path.",
    )
    p.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold.")
    p.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config options like detectron2: KEY VALUE ...",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Make relative paths behave as expected when running from repo root.
    os.chdir(str(_REPO_ROOT))

    cfg, class_names = setup_cfg(args)
    img, outputs = infer_one_image(cfg, args.input)
    draw_and_save(img, outputs, class_names, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
