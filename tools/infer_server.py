import argparse
import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

# Ensure `second_14.1` repo root is importable when running as:
#   python tools/infer_server.py ...
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


def build_predictor(config_file: str, weights: str, score_thresh: float, opts) -> Tuple[DefaultPredictor, list]:
    cfg = get_cfg()
    add_ovd_config(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(score_thresh)

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
    predictor = DefaultPredictor(cfg)
    return predictor, open_vocab_classes


def decode_image_bgr(image_bytes: bytes):
    arr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("Failed to decode image bytes.")
    return arr


def encode_jpg_b64(img_bgr, quality: int = 95) -> str:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", img_bgr, params)
    if not ok:
        raise OSError("cv2.imencode(.jpg) failed.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def instances_to_json(instances, class_names):
    if instances is None:
        return []
    inst = instances.to("cpu")
    boxes = inst.pred_boxes.tensor.tolist() if inst.has("pred_boxes") else []
    scores = inst.scores.tolist() if inst.has("scores") else []
    classes = inst.pred_classes.tolist() if inst.has("pred_classes") else []
    out = []
    for box, score, cls_id in zip(boxes, scores, classes):
        name = None
        if class_names and 0 <= int(cls_id) < len(class_names):
            name = class_names[int(cls_id)]
        out.append(
            {
                "bbox_xyxy": [float(x) for x in box],
                "score": float(score),
                "class_id": int(cls_id),
                "class_name": name,
            }
        )
    return out


def draw_predictions(img_bgr, instances, class_names):
    metadata = MetadataCatalog.get("open_vocab_infer")
    if class_names:
        metadata.set(thing_classes=list(class_names))
    viz = Visualizer(img_bgr[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    vis_img = viz.draw_instance_predictions(instances.to("cpu")).get_image()[:, :, ::-1]
    return vis_img


class InferApp:
    def __init__(self, predictor: DefaultPredictor, class_names: list):
        self.predictor = predictor
        self.class_names = class_names

    def infer(self, image_bytes: bytes, score_thresh: Optional[float] = None) -> Dict[str, Any]:
        img_bgr = decode_image_bgr(image_bytes)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.predictor(img_bgr)
        t_infer = time.perf_counter() - t0

        inst = outputs.get("instances", None)
        if inst is not None and score_thresh is not None and inst.has("scores"):
            keep = inst.scores > float(score_thresh)
            inst = inst[keep]

        t1 = time.perf_counter()
        vis = draw_predictions(img_bgr, inst, self.class_names) if inst is not None else img_bgr
        t_draw = time.perf_counter() - t1

        return {
            "detections": instances_to_json(inst, self.class_names),
            "vis_jpg_b64": encode_jpg_b64(vis),
            "timing": {"infer_s": t_infer, "draw_s": t_draw},
        }


def make_handler(app: InferApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return  # quiet

        def _send_json(self, code: int, payload: Dict[str, Any]):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"ok": False, "error": "Not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/infer":
                self._send_json(404, {"ok": False, "error": "Not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8"))
                image_b64 = req.get("image_b64", None)
                if not image_b64:
                    raise ValueError("Missing field: image_b64")
                image_bytes = base64.b64decode(image_b64)
                score_thresh = req.get("score_thresh", None)
                result = app.infer(image_bytes=image_bytes, score_thresh=score_thresh)
                self._send_json(200, {"ok": True, "result": result})
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})

    return Handler


def parse_args():
    p = argparse.ArgumentParser(description="HTTP single-image inference service.")
    p.add_argument(
        "--config-file",
        default="configs/coco/COCO_OVD_Food2K_PIS.yaml",
        help="Config yaml path (relative to second_14.1/ or absolute).",
    )
    p.add_argument(
        "--weights",
        default="output/coco_ovd_food2k_PIS_with_cap_new/model_0079999.pth",
        help="Weights path (relative to second_14.1/ or absolute).",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host (use 127.0.0.1 with SSH port-forward).")
    p.add_argument("--port", type=int, default=18080, help="Bind port on the server.")
    p.add_argument("--score-thresh", type=float, default=0.5, help="Default score threshold.")
    p.add_argument("opts", nargs=argparse.REMAINDER, help="Extra cfg opts: KEY VALUE ...")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(str(_REPO_ROOT))

    predictor, class_names = build_predictor(
        config_file=args.config_file,
        weights=args.weights,
        score_thresh=args.score_thresh,
        opts=args.opts,
    )
    app = InferApp(predictor=predictor, class_names=class_names)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"[infer_server] listening on http://{args.host}:{args.port}")
    print("[infer_server] health: GET /health")
    print("[infer_server] infer:  POST /infer  (json: {image_b64, score_thresh?})")
    server.serve_forever()


if __name__ == "__main__":
    main()
