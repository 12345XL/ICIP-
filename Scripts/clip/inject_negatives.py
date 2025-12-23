"""
# 文件说明（clip/inject_negatives_dino.py）

- **文件作用**：与 `inject_negatives.py` 类似，为 CLIP 向量库注入 CutPaste 伪缺陷，但增加了特征维度一致性检查。
- **运行方式**：在项目根目录执行 `python Scripts/clip/inject_negatives_dino.py`，适用于已经存在 `embeddings/` 的场景。
- **输出结果**：在 `embeddings/` 目录下的 Patch 索引中追加 `type='synthetic'` 的负样本 Patch。
- **分类角色**：归属于 `clip` 分类，是 CLIP 管线的「增强版负样本注入」脚本。
"""

import os
import clip
import torch
import faiss
import pickle
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

# --- 配置 (必须与你 build_visual_bank.py 保持一致) ---
DATASET_ROOT = "/data/XL/多模态RAG/DataSet/MVTec-AD"
EMBEDDING_DIR = "./embeddings"  # 与构建索引的目录保持一致
MODEL_NAME = "ViT-L/14@336px"
STRIDE_RATIO = 0.5
SYNTHETIC_RATIO = 0.5  # 每2张正常图，就生成1张图量的“假缺陷”

class NegativeInjector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading CLIP ({MODEL_NAME}) on {self.device}...")
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)
        self.input_size = self.model.visual.input_resolution
        self.stride = int(self.input_size * STRIDE_RATIO)

    def generate_synthetic_defect(self, img_pil):
        """核心：CutPaste 制造人造缺陷"""
        w, h = img_pil.size
        # 1. 随机切一个小 Patch (5%-15% 尺寸)
        patch_w = random.randint(int(w*0.05), int(w*0.15))
        patch_h = random.randint(int(h*0.05), int(h*0.15))
        
        src_x = random.randint(0, w - patch_w)
        src_y = random.randint(0, h - patch_h)
        patch = img_pil.crop((src_x, src_y, src_x + patch_w, src_y + patch_h))
        
        # 2. 坏样增强 (变色、变亮、色调分离，模拟异物/划痕)
        if random.random() > 0.5: patch = ImageOps.posterize(patch, 4)
        patch = ImageEnhance.Color(patch).enhance(random.uniform(0.5, 2.5))
        patch = ImageEnhance.Brightness(patch).enhance(random.uniform(0.6, 1.4))
        
        # 3. 贴回去 (随机位置)
        dst_x = random.randint(0, w - patch_w)
        dst_y = random.randint(0, h - patch_h)
        aug_img = img_pil.copy()
        aug_img.paste(patch, (dst_x, dst_y))
        
        return aug_img

    def process_category(self, category):
        print(f"\n⚡ Injecting Negatives for: {category}")
        
        patch_index_path = os.path.join(EMBEDDING_DIR, f"{category}_patch.index")
        meta_path = os.path.join(EMBEDDING_DIR, f"{category}_meta.pkl")
        
        if not os.path.exists(patch_index_path):
            return

        # 1. 加载现有索引
        index = faiss.read_index(patch_index_path)
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f) # 格式: {'global_paths': [], 'patch_info': []}
        
        print(f"   -> Original Size: {index.ntotal}")

        # 2. 扫描并生成
        train_good_path = os.path.join(DATASET_ROOT, category, "train", "good")
        img_files = [f for f in os.listdir(train_good_path) if f.lower().endswith(('.png', '.jpg'))]
        
        new_vectors = []
        new_patch_meta = []
        
        for img_idx, img_file in enumerate(tqdm(img_files, desc="Generating Synthetics")):
            if random.random() > SYNTHETIC_RATIO: continue # 按比例抽样
            
            img_path = os.path.join(train_good_path, img_file)
            try:
                pil_img = Image.open(img_path).convert("RGB")
                bad_img = self.generate_synthetic_defect(pil_img) # 造假
                
                # 切片并提取特征
                w, h = bad_img.size
                batch_patches = []
                batch_coords = []
                
                for y in range(0, h - self.input_size + 1, self.stride):
                    for x in range(0, w - self.input_size + 1, self.stride):
                        box = (x, y, x + self.input_size, y + self.input_size)
                        patch = self.preprocess(bad_img.crop(box))
                        batch_patches.append(patch)
                        batch_coords.append((x, y))
                
                if batch_patches:
                    tensor = torch.stack(batch_patches).to(self.device)
                    with torch.no_grad():
                        feats = self.model.encode_image(tensor)
                        feats /= feats.norm(dim=-1, keepdim=True)
                    
                    new_vectors.append(feats.cpu().numpy())
                    
                    # 记录元数据，关键是 type='synthetic'
                    for (x, y) in batch_coords:
                        new_patch_meta.append({
                            "parent_idx": -1, # 负样本不关联具体Global图
                            "coords": (x, y),
                            "type": "synthetic" 
                        })
            except Exception as e:
                print(f"Error: {e}")

        if not new_vectors:
            return

        # 3. 追加并保存
        final_vectors = np.concatenate(new_vectors, axis=0).astype('float32')
        if final_vectors.shape[1] != index.d:
            raise ValueError(f"Embedding dim mismatch: vectors {final_vectors.shape[1]} vs index {index.d}. 请确保模型与索引来源一致")
        index.add(final_vectors)
        
        # 给旧数据补上 type 标记
        for item in meta_data['patch_info']:
            if 'type' not in item: item['type'] = 'normal'
            
        meta_data['patch_info'].extend(new_patch_meta)
        
        faiss.write_index(index, patch_index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_data, f)
            
        print(f"✅ Added {len(new_patch_meta)} negative patches. New Size: {index.ntotal}")

if __name__ == "__main__":
    injector = NegativeInjector()
    categories = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    for cat in categories:
        injector.process_category(cat)
