"""
自动化脚本：读取 Food2K 类别，与 ZSFooD/COCO 未见类的零样本文本嵌入计算相似度。
结果以“每个未见类 TopK 对应的 Food2K 类别”形式输出。
运行方式：python tools/test_class_similarity.py
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import clip

import ovd  # noqa: F401  # 保证 Detectron2 Dataset 已注册
from ovd.datasets.coco_zeroshot import categories_unseen
from detectron2.data import MetadataCatalog

FOOD2K_JSON = Path("/22liushoulong/projects5/second/process_food2k_pas_coco/food2k_train.json")
ZS_WEIGHT_PATH = Path("/22liushoulong/projects5/second/datasets/zeroshot_weights/coco_clip_a+photo+cname_o.npy")
BASE_DATASET = "coco_generalized_zeroshot_val"
CLIP_MODEL = "ViT-B/32"
TOPK_PER_UNSEEN = 44
TOPK_FOOD2K = 50
OUTPUT_PATH = Path("output/food2k_similarity.json")
BATCH_SIZE = 64


def get_base_class_names(dataset_name: str) -> List[str]:
    meta = MetadataCatalog.get(dataset_name)
    if hasattr(meta, "open_vocab_classes") and len(meta.open_vocab_classes) > 0:
        return list(meta.open_vocab_classes)
    return list(meta.thing_classes)


def get_unseen_class_info(dataset_name: str):
    """
    返回 base dataset 中未见类名称以及它们在 zeroshot 权重里的下标。
    """
    base_names = get_base_class_names(dataset_name)
    unseen_name_set = {c["name"] for c in categories_unseen}
    indices = [idx for idx, name in enumerate(base_names) if name in unseen_name_set]
    filtered_names = [base_names[idx] for idx in indices]
    missing = unseen_name_set - set(filtered_names)
    if missing:
        print(f"[WARN] {len(missing)} unseen classes not found in metadata/weights: {sorted(missing)}")
    assert len(indices) > 0, "未找到任何未见类名称，请检查 coco_zeroshot 配置。"
    return filtered_names, indices


def load_food2k_classes(json_path: Path) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data.get("food2k_classes", [])
    return list(classes)


def chunk_list(values: List[str], chunk_size: int):
    for i in range(0, len(values), chunk_size):
        yield values[i:i + chunk_size]


def main():
    assert FOOD2K_JSON.is_file(), f"Food2K json not found: {FOOD2K_JSON}"
    assert ZS_WEIGHT_PATH.is_file(), f"Zero-shot weight file not found: {ZS_WEIGHT_PATH}"

    food2k_classes = load_food2k_classes(FOOD2K_JSON)
    unseen_class_names, unseen_indices = get_unseen_class_info(BASE_DATASET)

    base_embeddings_np = np.load(ZS_WEIGHT_PATH)
    if base_embeddings_np.shape[0] < max(unseen_indices) + 1:
        raise ValueError("Zeroshot 权重文件条目不足，无法索引到所有未见类。")
    # 仅拿出未见类对应的嵌入
    base_embeddings_np = base_embeddings_np[unseen_indices]
    if base_embeddings_np.shape[0] != len(unseen_class_names):
        min_len = min(base_embeddings_np.shape[0], len(unseen_class_names))
        print(f"[WARN] Zeroshot weights ({base_embeddings_np.shape[0]}) "
              f"do not match class names ({len(unseen_class_names)}); "
              f"truncating to first {min_len}.")
        base_embeddings_np = base_embeddings_np[:min_len]
        unseen_class_names = unseen_class_names[:min_len]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(CLIP_MODEL, device=device)
    base_embeddings = torch.from_numpy(base_embeddings_np).float().to(device)
    base_embeddings = torch.nn.functional.normalize(base_embeddings, dim=1)
    encoded_food_feats = []
    encoded_food_names = []

    with torch.no_grad():
        for chunk in chunk_list(food2k_classes, BATCH_SIZE):
            prompts = [f"a photo of {name}" for name in chunk]
            token_ids = clip.tokenize(prompts).to(device)
            feats = clip_model.encode_text(token_ids)
            feats = torch.nn.functional.normalize(feats, dim=-1).to(base_embeddings.dtype)
            encoded_food_feats.append(feats)
            encoded_food_names.extend(chunk)

    food_embeddings = torch.cat(encoded_food_feats, dim=0)
    assert len(encoded_food_names) == len(food2k_classes)

    # 计算相似度矩阵
    sims = base_embeddings @ food_embeddings.T

    # 每个未见类 TopK 对应的 Food2K 类
    per_unseen_values, per_unseen_indices = sims.topk(
        k=min(TOPK_PER_UNSEEN, food_embeddings.shape[0]), dim=1)
    per_unseen_results = []
    for base_name, scores, inds in zip(
            unseen_class_names, per_unseen_values.cpu(), per_unseen_indices.cpu()):
        matches = [
            {"food2k_class": encoded_food_names[idx], "score": float(score)}
            for idx, score in zip(inds.tolist(), scores.tolist())
        ]
        per_unseen_results.append({"base_class": base_name, "matches": matches})

    # 在 1944*44 的相似度分数中找出最高的 Food2K 类别
    best_scores_per_food, best_unseen_idx = sims.max(dim=0)
    topk_scores, topk_food_indices = best_scores_per_food.topk(
        k=min(TOPK_FOOD2K, best_scores_per_food.numel()))
    top_food2k = []
    for food_idx, score in zip(topk_food_indices.cpu().tolist(), topk_scores.cpu().tolist()):
        unseen_idx = best_unseen_idx[food_idx].item()
        top_food2k.append({
            "food2k_class": encoded_food_names[food_idx],
            "best_unseen_class": unseen_class_names[unseen_idx],
            "score": float(score),
        })

    payload = {
        "per_unseen_results": per_unseen_results,
        "top_food2k": top_food2k,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Processed {len(unseen_class_names)} unseen base classes.")
    print(f"[INFO] Food2K pool size: {len(encoded_food_names)}")
    print(f"[INFO] Full results saved to {OUTPUT_PATH}")
    print("[Preview] 未见类 -> Top1 Food2K")
    for entry in per_unseen_results[:TOPK_PER_UNSEEN]:
        top_match = entry["matches"][0]
        print(f"- {entry['base_class']} -> {top_match['food2k_class']} (sim={top_match['score']:.4f})")
    print("[Preview] Food2K Top scores")
    for entry in top_food2k[:TOPK_PER_UNSEEN]:
        print(f"- {entry['food2k_class']} (sim={entry['score']:.4f}) via {entry['best_unseen_class']}")


if __name__ == "__main__":
    main()
