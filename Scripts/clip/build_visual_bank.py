"""
# 文件说明（clip/build_visual_bank.py）

- **文件作用**：使用 CLIP 将训练集正常图片编码为全局特征与 Patch 特征，并构建分层 FAISS 向量库。
- **运行方式**：在项目根目录执行 `python Scripts/clip/build_visual_bank.py`，会自动遍历 `DATASET_ROOT` 下所有类别。
- **输出结果**：在当前工作目录生成 `embeddings/` 目录，包含 `<category>_global.index`、`<category>_patch.index` 和 `*_meta.pkl`。
- **分类角色**：归属于 `clip` 分类，是 CLIP 视觉检索管线中的「索引构建」脚本。
"""

import os
import clip
import torch
import faiss
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 配置参数 ---
DATASET_ROOT = "/data/XL/多模态RAG/DataSet/MVTec-AD"
OUTPUT_DIR = "./embeddings"
MODEL_NAME = "ViT-L/14@336px"
STRIDE_RATIO = 0.5
BATCH_SIZE = 64

class HierarchicalBankBuilder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading CLIP ({MODEL_NAME}) on {self.device}...")
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)
        self.input_size = self.model.visual.input_resolution
        self.stride = int(self.input_size * STRIDE_RATIO)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def process_category(self, category):
        """处理单个类别的核心逻辑"""
        print(f"\n⚡ Processing Category: {category}")
        train_good_path = os.path.join(DATASET_ROOT, category, "train", "good")
        
        if not os.path.exists(train_good_path):
            return

        # 容器
        global_features = []   # 存整图特征 (用于阶段1检索)
        patch_features = []    # 存Patch特征 (用于阶段2对齐)
        
        # 索引映射
        global_metadata = []   # {img_path}
        patch_metadata = []    # {parent_img_idx, coords} -> parent_img_idx 指向 global_metadata

        img_files = [f for f in os.listdir(train_good_path) if f.lower().endswith(('.png', '.jpg'))]
        
        for img_idx, img_file in enumerate(tqdm(img_files, desc=f"Encoding {category}")):
            img_path = os.path.join(train_good_path, img_file)
            
            try:
                # --- 1. 读取与全局特征 ---
                pil_img = Image.open(img_path).convert("RGB")
                global_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    g_feat = self.model.encode_image(global_input)
                    g_feat /= g_feat.norm(dim=-1, keepdim=True)
                
                global_features.append(g_feat.cpu().numpy())
                global_metadata.append(img_path)
                
                # --- 2. 提取 Patches ---
                w, h = pil_img.size
                batch_patches = []
                batch_coords = []
                
                for y in range(0, h - self.input_size + 1, self.stride):
                    for x in range(0, w - self.input_size + 1, self.stride):
                        box = (x, y, x + self.input_size, y + self.input_size)
                        patch = pil_img.crop(box)
                        batch_patches.append(self.preprocess(patch))
                        batch_coords.append((x, y))
                
                if batch_patches:
                    patch_tensor = torch.stack(batch_patches).to(self.device)
                    with torch.no_grad():
                        p_feat = self.model.encode_image(patch_tensor)
                        p_feat /= p_feat.norm(dim=-1, keepdim=True)
                    
                    patch_features.append(p_feat.cpu().numpy())
                    
                    # 记录 Patch 的“父级”是谁，以及坐标
                    for (x, y) in batch_coords:
                        patch_metadata.append({
                            "parent_idx": img_idx, # 关联到 global_metadata[img_idx]
                            "coords": (x, y)
                        })
                        
            except Exception as e:
                print(f"Error: {e}")

        # --- 3. 存盘 (分层存储) ---
        if global_features:
            # A. 全局索引
            g_emb = np.concatenate(global_features, axis=0).astype('float32')
            g_index = faiss.IndexFlatIP(g_emb.shape[1])
            g_index.add(g_emb)
            faiss.write_index(g_index, os.path.join(OUTPUT_DIR, f"{category}_global.index"))
            
            # B. 局部 Patch 索引
            p_emb = np.concatenate(patch_features, axis=0).astype('float32')
            p_index = faiss.IndexFlatIP(p_emb.shape[1])
            p_index.add(p_emb)
            faiss.write_index(p_index, os.path.join(OUTPUT_DIR, f"{category}_patch.index"))
            
            # C. 元数据
            with open(os.path.join(OUTPUT_DIR, f"{category}_meta.pkl"), 'wb') as f:
                pickle.dump({
                    "global_paths": global_metadata,
                    "patch_info": patch_metadata
                }, f)
            
            print(f"✅ {category}: {g_index.ntotal} images, {p_index.ntotal} patches.")

if __name__ == "__main__":
    builder = HierarchicalBankBuilder()
    categories = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    for cat in categories:
        builder.process_category(cat)
