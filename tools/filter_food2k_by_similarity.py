"""
根据未见类与 Food2K 类别之间的 CLIP 相似度，保留最相似的前 TOP_K Food2K 类别并过滤数据集。
运行方式：python tools/filter_food2k_by_similarity.py
"""

import json
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import torch
import clip

import ovd  # noqa: F401  # 确保 Detectron2 数据集注册
from ovd.datasets.coco_zeroshot import categories_unseen
from detectron2.data import MetadataCatalog

FOOD2K_JSON = Path("/22liushoulong/projects5/second/process_food2k_pas_coco/food2k_train.json")
OUTPUT_JSON = Path("/22liushoulong/projects5/second/process_food2k_pas_coco/food2k_train_filtered.json")
ZS_WEIGHT_PATH = Path("/22liushoulong/projects5/second/datasets/zeroshot_weights/coco_clip_a+photo+cname_o.npy")
BASE_DATASET = "coco_generalized_zeroshot_val"
CLIP_MODEL = "ViT-B/32"
TOP_K_CLASSES = 100
BATCH_SIZE = 64


def get_base_class_names(dataset_name: str) -> List[str]:
    meta = MetadataCatalog.get(dataset_name)
    if hasattr(meta, "open_vocab_classes") and len(meta.open_vocab_classes) > 0:
        return list(meta.open_vocab_classes)
    return list(meta.thing_classes)


def get_unseen_class_info(dataset_name: str):
    base_names = get_base_class_names(dataset_name)
    unseen_name_set = {c["name"] for c in categories_unseen}
    indices = [idx for idx, name in enumerate(base_names) if name in unseen_name_set]
    filtered_names = [base_names[idx] for idx in indices]
    missing = unseen_name_set - set(filtered_names)
    if missing:
        print(f"[WARN] {len(missing)} unseen classes not found in metadata/weights: {sorted(missing)}")
    assert len(indices) > 0, "未找到任何未见类，请检查 coco_zeroshot 配置。"
    return filtered_names, indices


def load_food2k_classes(json_path: Path) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("food2k_classes", []))


def chunk_list(values: List[str], chunk_size: int):
    for i in range(0, len(values), chunk_size):
        yield values[i:i + chunk_size]


def encode_food_classes(food_classes: List[str], device: str, clip_model) -> torch.Tensor:
    feats = []
    for chunk in chunk_list(food_classes, BATCH_SIZE):
        prompts = [f"a photo of {name}" for name in chunk]
        token_ids = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(token_ids)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        feats.append(emb)
    return torch.cat(feats, dim=0)


def select_food2k_classes(food_classes: List[str]):
    unseen_class_names, unseen_indices = get_unseen_class_info(BASE_DATASET)
    base_embeddings_np = np.load(ZS_WEIGHT_PATH)
    if base_embeddings_np.shape[0] < max(unseen_indices) + 1:
        raise ValueError("Zeroshot 权重文件条目不足，无法索引到所有未见类。")
    base_embeddings_np = base_embeddings_np[unseen_indices]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(CLIP_MODEL, device=device)
    base_embeddings = torch.from_numpy(base_embeddings_np).float().to(device)
    base_embeddings = torch.nn.functional.normalize(base_embeddings, dim=1)

    food_embeddings = encode_food_classes(food_classes, device, clip_model).to(device).to(base_embeddings.dtype)

    sims = base_embeddings @ food_embeddings.T  # [num_unseen, num_food]

    # 针对每个 Food2K 类别，找到与所有未见类中最高的相似度
    per_food_scores, best_unseen_indices = sims.max(dim=0)
    top_k = min(TOP_K_CLASSES, per_food_scores.numel())
    top_scores, top_food_indices = torch.topk(per_food_scores, k=top_k)

    selected_pairs = []
    selected_food_names: Set[str] = set()
    for score, food_idx in zip(top_scores.cpu().tolist(), top_food_indices.cpu().tolist()):
        unseen_idx = best_unseen_indices[food_idx].item()
        food_name = food_classes[food_idx]
        unseen_name = unseen_class_names[unseen_idx]
        selected_pairs.append({
            "food2k_class": food_name,
            "unseen_class": unseen_name,
            "score": float(score),
        })
        selected_food_names.add(food_name)

    print(f"[INFO] 根据未见类相似度保留前 {len(selected_food_names)} 个 Food2K 类别（TOP_K={TOP_K_CLASSES}）。")
    for pair in selected_pairs:
        food = pair["food2k_class"]
        unseen = pair["unseen_class"]
        score = pair["score"]
        print(f"  - {food:<40} <-> {unseen:<30} score={score:.4f}")
    return selected_food_names, selected_pairs


def filter_food2k_json(selected_food_names: Set[str], match_pairs, output_path: Path):
    with open(FOOD2K_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_classes = data.get("food2k_classes", [])

    kept_classes = [cls for cls in original_classes if cls in selected_food_names]
    keep_set = set(kept_classes)
    new_class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(kept_classes)}

    filtered_records = []
    for record in data.get("records", []):
        cls_name = record.get("class_name")
        if cls_name in keep_set:
            record_copy = record.copy()
            new_idx = new_class_to_idx[cls_name]
            record_copy["image_level_labels"] = [new_idx]
            filtered_records.append(record_copy)

    filtered_data = {
        "records": filtered_records,
        "food2k_classes": kept_classes,
        "similarity_matches": match_pairs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 保存至 {output_path}。保留 {len(kept_classes)} 个类别、{len(filtered_records)} 张图像。")


def main():
    assert FOOD2K_JSON.is_file(), f"Food2K json not found: {FOOD2K_JSON}"
    assert ZS_WEIGHT_PATH.is_file(), f"Zero-shot weight file not found: {ZS_WEIGHT_PATH}"

    food2k_classes = load_food2k_classes(FOOD2K_JSON)
    selected_food_names, match_pairs = select_food2k_classes(food2k_classes)
    if not selected_food_names:
        print("[WARN] 没有筛选到任何 Food2K 类别，未生成过滤文件。")
        return
    filter_food2k_json(selected_food_names, match_pairs, OUTPUT_JSON)


if __name__ == "__main__":
    main()
