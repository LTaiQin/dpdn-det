# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

# --- 配置区：已根据您的要求更新 ---

# 1. 输入文件路径
DETECTION_RESULTS_FILE = "/22liushoulong/projects4/ovfoodD_simple_origin_prompt/output/coco_ovd_PIS_origin/inference_coco_generalized_zeroshot_val/coco_instances_results.json"
GT_ANNOTATION_FILE = "/22liushoulong/datasets/ZSFooD2/annotations/instances_val2017.json"
IMAGE_DIRECTORY = "/22liushoulong/datasets/ZSFooD2/val2017/"

# 2. 输出文件路径
OUTPUT_DIRECTORY = "/22liushoulong/datasets/ZSFooD2/val_image_boxed_ORIGIN_PROMPT_parallel/"

# 3. 筛选参数
CONFIDENCE_THRESHOLD = 0.5  # 筛选阈值：只显示分数高于 0.5 的检测框
TOP_K_PER_IMAGE = 10      # 每张图片最多显示的检测框数量（按分数排序）

# 4. 并行处理参数
# 设置为 0 或 None 将使用所有可用的 CPU 核心
# 您也可以指定一个具体的数字，例如 8
MAX_WORKERS = None 


# --- 核心功能代码 ---

def get_color_palette(num_categories):
    """为不同类别生成一组区分度较高的颜色"""
    palette = []
    for i in range(num_categories):
        hue = (i * 0.618033988749895) % 1.0
        palette.append(plt.cm.hsv(hue))
    return palette


def process_and_save_image(image_item, image_dir_path, output_dir_path, category_map,
                           category_colors, confidence_threshold, top_k, image_map):
    """
    处理单张图片：筛选检测框、加载图片、绘制边界框并保存。
    此函数被设计为在单独的进程中运行。
    """
    image_id, image_detections = image_item

    # --- 筛选步骤 ---
    high_conf_dets = [d for d in image_detections if d['score'] >= confidence_threshold]
    high_conf_dets.sort(key=lambda x: x['score'], reverse=True)
    if top_k > 0:
        final_dets_to_draw = high_conf_dets[:top_k]
    else:
        final_dets_to_draw = high_conf_dets

    if not final_dets_to_draw:
        return  # 如果筛选后没有框，则跳过这张图

    # --- 绘图步骤 ---
    image_filename = image_map.get(image_id)
    if not image_filename:
        print(f"警告: 图像ID {image_id} 在标注文件中未找到对应文件名，跳过。")
        return

    image_path = os.path.join(image_dir_path, image_filename)
    if not os.path.exists(image_path):
        print(f"警告: 图像文件不存在: {image_path}，跳过。")
        return

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"错误: 无法打开图像 {image_path}: {e}")
        return

    fig, ax = plt.subplots(1, figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)

    for det in final_dets_to_draw:
        bbox = det['bbox']
        category_id = det['category_id']
        color = category_colors.get(category_id, 'red')  # 默认红色

        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

    ax.axis('off')
    fig.tight_layout(pad=0)
    output_filename = os.path.join(output_dir_path, image_filename)
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def visualize_all_detections_parallel(
        json_results_path,
        gt_annotation_path,
        image_dir_path,
        output_dir_path,
        confidence_threshold=0.5,
        top_k=10,
        max_workers=None
):
    """
    加载检测结果，并使用多核并行处理来可视化所有图像。
    """
    # 1. 创建输出目录
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"创建输出目录: {output_dir_path}")

    # 2. 加载数据文件
    print("正在加载检测结果文件...")
    try:
        with open(json_results_path, 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        print(f"错误: 检测结果文件未找到: {json_results_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件: {json_results_path}")
        return

    print("正在加载标注文件以获取元数据...")
    try:
        with open(gt_annotation_path, 'r') as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 标注文件未找到: {gt_annotation_path}")
        return

    # 3. 创建数据映射
    category_map = {cat['id']: cat['name'] for cat in gt_data.get('categories', [])}
    image_map = {img['id']: img['file_name'] for img in gt_data.get('images', [])}

    if not image_map:
        print("错误: 标注文件中未找到 'images' 信息。无法匹配图像文件。")
        return

    # 4. 按图像ID对检测结果进行分组
    detections_by_image = defaultdict(list)
    for det in detections:
        detections_by_image[det['image_id']].append(det)

    if not detections_by_image:
        print("错误: 检测结果为空或格式不正确。")
        return

    # 5. 生成颜色调色板
    colors = get_color_palette(len(category_map) if category_map else 20)
    category_colors = {cat_id: colors[i % len(colors)] for i, cat_id in enumerate(category_map.keys())}
    
    # 6. 设置并行处理
    # 使用 partial 预先绑定那些在所有任务中都保持不变的参数
    worker_function = partial(
        process_and_save_image,
        image_dir_path=image_dir_path,
        output_dir_path=output_dir_path,
        category_map=category_map,
        category_colors=category_colors,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
        image_map=image_map
    )

    tasks = list(detections_by_image.items())
    num_tasks = len(tasks)
    
    print(f"发现 {num_tasks} 张图像有检测结果，将使用多核并行处理...")
    
    # 使用 ProcessPoolExecutor 进行并行处理，并用 tqdm 显示进度条
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 list(...) 来确保所有任务都完成
        list(tqdm(executor.map(worker_function, tasks), total=num_tasks, desc="处理图像中"))

    print(f"\n处理完成！所有图像的可视化结果已保存至: {output_dir_path}")


# --- 脚本执行入口 ---
if __name__ == "__main__":
    visualize_all_detections_parallel(
        json_results_path=DETECTION_RESULTS_FILE,
        gt_annotation_path=GT_ANNOTATION_FILE,
        image_dir_path=IMAGE_DIRECTORY,
        output_dir_path=OUTPUT_DIRECTORY,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        top_k=TOP_K_PER_IMAGE,
        max_workers=MAX_WORKERS
    )