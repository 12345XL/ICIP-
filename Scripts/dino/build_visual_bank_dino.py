"""
# 文件说明（dino/build_visual_bank_dino.py）

- **文件作用**：遍历 MVTec-AD 训练集正常图片，使用 DINOv2 提取特征并构建 FAISS 向量库。
- **运行方式**：在项目根目录执行 `python Scripts/dino/build_visual_bank_dino.py`，确保数据集路径与依赖安装正确。
- **输出结果**：在当前工作目录生成 `embeddings_dino/` 目录，内含每个类别的 `_global.index`、`_patch.index` 和 `*_meta.pkl`。
- **分类角色**：归属于 `dino` 分类，是 DINO 视觉检索管线中的「索引构建」脚本。
"""

import os
import sys
import faiss
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPTS_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.append(SCRIPTS_DIR)
sys.path.append(PROJECT_ROOT)
from feature_extractor import FeatureExtractor
from config import DATASET_ROOT
from utils.paths import make_index_id, index_root, category_dir

STRIDE_RATIO = 0.5
TOPK = 5

class DINOVisualBankBuilder:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.input_size = self.extractor.input_size
        self.stride = int(self.input_size * STRIDE_RATIO)
        backbone = "dinov2"
        env_index_id = os.environ.get("INDEX_ID")
        if env_index_id:
            self.index_id = env_index_id
        else:
            self.index_id = make_index_id(
                backbone=backbone,
                input_size=self.input_size,
                stride=self.stride,
                topk=TOPK,
                extra="normal",
            )
        self.root = index_root(self.index_id)
        print(f"Using index_id = {self.index_id}, root = {self.root}")

    def process_category(self, category):
        print(f"\n⚡ Processing Category: {category}")
        train_good_path = os.path.join(DATASET_ROOT, category, "train", "good")
        if not os.path.exists(train_good_path): return

        patch_features = []
        global_features = [] 
        global_metadata = []
        patch_metadata = []

        img_files = [f for f in os.listdir(train_good_path) if f.lower().endswith(('.png', '.jpg'))]
        
        for img_idx, img_file in enumerate(tqdm(img_files, desc=f"Encoding {category}")):
            img_path = os.path.join(train_good_path, img_file)
            try:
                pil_img = Image.open(img_path).convert("RGB")
                
                # 1. 全局特征 (用于找参照图)
                g_input = [self.extractor.preprocess(pil_img)]
                g_feat = self.extractor.encode(g_input)
                global_features.append(g_feat)
                global_metadata.append(img_path)
                
                w, h = pil_img.size
                batch_patches = []
                batch_coords = []
                
                ys = list(range(0, h - self.extractor.input_size + 1, self.stride))
                xs = list(range(0, w - self.extractor.input_size + 1, self.stride))
                if ys and ys[-1] != h - self.extractor.input_size:
                    ys.append(h - self.extractor.input_size)
                if xs and xs[-1] != w - self.extractor.input_size:
                    xs.append(w - self.extractor.input_size)

                for y in ys:
                    for x in xs:
                        box = (x, y, x + self.extractor.input_size, y + self.extractor.input_size)
                        patch = pil_img.crop(box)
                        batch_patches.append(self.extractor.preprocess(patch))
                        batch_coords.append((x, y))
                
                if batch_patches:
                    p_feats = self.extractor.encode(batch_patches)
                    patch_features.append(p_feats)
                    for (x, y) in batch_coords:
                        patch_metadata.append({
                            "parent_idx": img_idx,
                            "coords": (x, y),
                            "type": "normal"
                        })
            except Exception as e:
                print(f"Error: {e}")

        if patch_features:
            cat_dir = category_dir(self.root, category)
            patch_index_path = os.path.join(cat_dir, "patch.index")
            global_index_path = os.path.join(cat_dir, "global.index")
            meta_path = os.path.join(cat_dir, "meta.pkl")
            p_emb = np.concatenate(patch_features, axis=0)
            p_index = faiss.IndexFlatL2(p_emb.shape[1])
            p_index.add(p_emb)
            faiss.write_index(p_index, patch_index_path)
            g_emb = np.concatenate(global_features, axis=0)
            g_index = faiss.IndexFlatL2(g_emb.shape[1])
            g_index.add(g_emb)
            faiss.write_index(g_index, global_index_path)
            with open(meta_path, 'wb') as f:
                pickle.dump({"global_paths": global_metadata, "patch_info": patch_metadata}, f)
            print(f"✅ Saved DINOv2 index for {category} to {cat_dir}")

if __name__ == "__main__":
    builder = DINOVisualBankBuilder()
    env_cats = os.environ.get("TARGET_CATEGORIES", "").strip()
    if env_cats:
        categories = [c.strip() for c in env_cats.split(",") if c.strip()]
    else:
        categories = sorted(
            [
                d
                for d in os.listdir(DATASET_ROOT)
                if os.path.isdir(os.path.join(DATASET_ROOT, d))
            ]
        )
    for cat in categories:
        builder.process_category(cat)
