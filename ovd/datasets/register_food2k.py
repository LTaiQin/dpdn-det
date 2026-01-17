import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def load_food2k_from_json(
        json_path: str,
        class_name_to_contiguous: Optional[Dict[str, int]] = None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    class_map = None
    if class_name_to_contiguous is not None:
        class_map = {
            _normalize_name(k): v for k, v in class_name_to_contiguous.items()
        }

    for rec in data["records"]:
        rec = rec.copy()
        if class_map is not None:
            cls_name = _normalize_name(rec.get("class_name", ""))
            mapped_id = class_map.get(cls_name, None)
            if mapped_id is None:
                logger.warning(
                    "Food2K sample %s has class %s which is not mapped; skipping.",
                    rec.get("file_name", "unknown"),
                    rec.get("class_name", "unknown"),
                )
                continue
            rec["pos_category_ids"] = [mapped_id]
            rec["image_level_labels"] = [mapped_id]
            annotations = rec.get("annotations") or []
            for ann in annotations:
                ann.setdefault("bbox_mode", BoxMode.XYWH_ABS)
                ann.setdefault("category_id", mapped_id)
            rec["annotations"] = annotations
        records.append(rec)

    return records


def _build_class_union(
        reference_dataset: Optional[str],
        food2k_classes: List[str]) -> (List[str], Dict[str, int]):
    if reference_dataset is not None:
        ref_meta = MetadataCatalog.get(reference_dataset)
        base_classes = list(getattr(
            ref_meta,
            "open_vocab_classes",
            getattr(ref_meta, "thing_classes", [])))
    else:
        base_classes = []

    open_vocab_classes = list(base_classes)
    class_map = {_normalize_name(name): idx for idx, name in enumerate(open_vocab_classes)}

    for cls in food2k_classes:
        norm = _normalize_name(cls)
        if norm not in class_map:
            class_map[norm] = len(open_vocab_classes)
            open_vocab_classes.append(cls)

    return open_vocab_classes, class_map


def register_food2k_dataset(
        dataset_name: str,
        json_path: str,
        reference_dataset: Optional[str] = None):
    json_path = os.path.abspath(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    food2k_classes = data["food2k_classes"]

    open_vocab_classes, class_map = _build_class_union(reference_dataset, food2k_classes)

    DatasetCatalog.register(
        dataset_name,
        lambda: load_food2k_from_json(
            json_path,
            class_name_to_contiguous=class_map,
        )
    )

    metadata = MetadataCatalog.get(dataset_name)
    metadata.set(
        thing_classes=open_vocab_classes,
        food2k_classes=food2k_classes,
        open_vocab_classes=open_vocab_classes,
        json_file=json_path,
        reference_dataset=reference_dataset,
    )

    if reference_dataset is not None:
        try:
            ref_meta = MetadataCatalog.get(reference_dataset)
            ref_meta.set(open_vocab_classes=open_vocab_classes)
        except KeyError:
            logger.warning(
                "Reference dataset %s is not registered yet; open_vocab_classes not attached.",
                reference_dataset,
            )

    logger.info(
        "[INFO] Registered Food2K dataset: %s (reference=%s, total classes=%d)",
        dataset_name,
        reference_dataset or "None",
        len(open_vocab_classes),
    )


_DEFAULT_JSON = Path(__file__).resolve().parents[2] / "process_food2k_pas_coco" / "food2k_train_filtered.json"
if _DEFAULT_JSON.is_file():
    try:
        register_food2k_dataset(
            "food2k_train",
            str(_DEFAULT_JSON),
            reference_dataset="coco_zeroshot_train_oriorder",
        )
    except Exception as exc:
        logger.warning("Failed to auto-register Food2K dataset: %s", exc)
