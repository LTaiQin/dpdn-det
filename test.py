import dashscope
from http import HTTPStatus
import torch
import clip
from PIL import Image
import numpy as np

# --- 1. 配置 ---

# 1a. DashScope API 密钥 (来自您的代码)
dashscope.api_key = "sk-5bb016fd7e9d4e0ebb88f79db46edd6a"

# 1b. ！！请将这里替换为您的图片路径！！
IMAGE_PATH = "/22liushoulong/datasets/ZSFooD_filtered/zsfood_val_instances_clearned/braised drumstick with sauce/xsf_0_14__20201004_120924_317716.jpg"

# 1c. 共享的文本描述
TEXT_DESCRIPTION = "这是一盘红烧鸡腿配酱汁。"

# 1d. 设置 PyTorch 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("-" * 20)

# --- 2. (来自您的代码) 获取 DashScope 文本嵌入 ---

print("正在调用 DashScope API 获取文本嵌入...")
text_emb_dashscope_tensor = None
try:
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=TEXT_DESCRIPTION,
        dimension=512,  # 显式设置维度为 512
        output_type="dense"
    )

    if resp.status_code == HTTPStatus.OK:
        # 提取 512 维向量 (list)
        text_emb_dashscope_list = resp.output['embeddings'][0]['embedding']

        # 转换为 [1, 512] 的 PyTorch Tensor，并移动到目标 device
        text_emb_dashscope_tensor = torch.tensor(text_emb_dashscope_list).unsqueeze(0).to(device).float()
        print("DashScope 嵌入获取成功。")
    else:
        print(f"DashScope API 调用失败: {resp}")
        exit()
except Exception as e:
    print(f"DashScope API 调用时发生错误: {e}")
    exit()

print("-" * 20)

# --- 3. 加载和使用 CLIP (ViT-B/32) ---

print("正在加载 CLIP ViT-B/32 模型...")
try:
    # 加载 CLIP 模型和预处理器
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # 确保模型处于评估模式
    print("CLIP 模型加载成功。")
except Exception as e:
    print(f"加载 CLIP 模型失败。请确保已安装 'openai-clip': {e}")
    exit()

print("正在使用 CLIP 编码图片和文本...")
try:
    # 3a. 编码图片
    image = Image.open(IMAGE_PATH)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)  # [1, 512]

    # 3b. 编码文本
    text_input = clip.tokenize([TEXT_DESCRIPTION]).to(device)
    with torch.no_grad():
        text_features_clip = model.encode_text(text_input)  # [1, 512]

    print("CLIP 编码完成。")

except FileNotFoundError:
    print(f"错误: 找不到图片文件。请检查路径: {IMAGE_PATH}")
    exit()
except Exception as e:
    print(f"CLIP 编码过程中发生错误: {e}")
    exit()

print("-" * 20)

# --- 4. 归一化嵌入 (用于计算余弦相似度) ---

# 计算余弦相似度的标准做法是先对嵌入向量进行 L2 归一化
image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
text_features_clip_norm = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
# text_emb_dashscope_norm = text_emb_dashscope_tensor / text_emb_dashscope_tensor.norm(dim=-1, keepdim=True)
text_emb_dashscope_tensor = text_emb_dashscope_tensor.to(image_features_norm.dtype)
# --- 5. 计算并打印相似度 ---

# 相似度计算 (归一化后，向量点积即为余弦相似度)
similarity_img_clip = torch.mm(image_features_norm, text_features_clip_norm.T)
similarity_img_dashscope = torch.mm(image_features_norm, text_emb_dashscope_tensor.T)

print(f"--- 相似度计算结果 ---")
print(f"文本: '{TEXT_DESCRIPTION}'")
print(f"图片: {IMAGE_PATH}")
print("-" * 20)

# .item() 用于从
print(f"图片 (CLIP) vs. 文本 (CLIP ViT-B/32):     {similarity_img_clip.item():.4f}")
print(f"图片 (CLIP) vs. 文本 (DashScope v4): {similarity_img_dashscope.item():.4f}")