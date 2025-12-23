"""
# 文件说明（clip/inference.py）

- **文件作用**：基于 CLIP 向量库，对测试集图片生成带热力图叠加的可视化结果，用于调试 RAG 思路。
- **运行方式**：在项目根目录执行 `python Scripts/clip/inference.py`，可根据需要修改 `TARGET_CATEGORY`。
- **输出结果**：在 `results_visualization_v3/<category>/` 下生成 `[原图|参照图|热力图]` 拼接图片。
- **分类角色**：归属于 `clip` 分类，是 CLIP 视觉检索管线中的「推理与可视化」脚本。
"""

import os
import clip
import torch
import faiss
import pickle
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# --- 配置 ---
DATASET_ROOT = "/data/XL/多模态RAG/DataSet/MVTec-AD"
EMBEDDING_DIR = "./embeddings"
OUTPUT_DIR = "./results_visualization_v3" # 输出到新文件夹方便对比
MODEL_NAME = "ViT-L/14@336px"
STRIDE_RATIO = 0.5 
TARGET_CATEGORY = "bottle" 

class RAGInference:
    def __init__(self, category):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.category = category
        
        print(f"🚀 Loading CLIP on {self.device}...")
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)
        self.input_size = self.model.visual.input_resolution
        self.stride = int(self.input_size * STRIDE_RATIO)

        print(f"📂 Loading indices for {category}...")
        patch_index_path = os.path.join(EMBEDDING_DIR, f"{category}_patch.index")
        patch_meta_path = os.path.join(EMBEDDING_DIR, f"{category}_meta.pkl")
        
        self.patch_index = faiss.read_index(patch_index_path)
        with open(patch_meta_path, 'rb') as f:
            self.meta = pickle.load(f)
        self.patch_info = self.meta['patch_info']
        
        # Global Index (Optional)
        global_idx_path = os.path.join(EMBEDDING_DIR, f"{category}_global.index")
        if os.path.exists(global_idx_path):
            self.global_index = faiss.read_index(global_idx_path)
            self.global_paths = self.meta.get('global_paths', [])
        else:
            self.global_index = None

    def get_reference_image_path(self, img_feat):
        if self.global_index is None: return None
        D, I = self.global_index.search(img_feat, 1)
        ref_idx = I[0][0]
        if ref_idx < len(self.global_paths):
            return self.global_paths[ref_idx]
        return None

    def compute_anomaly_map(self, img_pil):
        w, h = img_pil.size
        patches = []
        coords = []
        for y in range(0, h - self.input_size + 1, self.stride):
            for x in range(0, w - self.input_size + 1, self.stride):
                box = (x, y, x + self.input_size, y + self.input_size)
                patches.append(self.preprocess(img_pil.crop(box)))
                coords.append((x, y))
        
        if not patches: return None, None

        # 1. 提取特征
        batch = torch.stack(patches).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats /= feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy().astype('float32')

        # 2. 检索 Top-K (K=5 取平均，更稳健)
        K = 5
        D, I = self.patch_index.search(feats_np, K)
        
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        for i, (x, y) in enumerate(coords):
            # 计算当前 Patch 的 Top-K 平均分
            k_scores = []
            for k in range(K):
                neighbor_idx = I[i][k]
                similarity = D[i][k] # Cosine Similarity (0-1)
                
                neighbor_type = self.patch_info[neighbor_idx].get('type', 'normal')
                
                # --- 改进后的打分逻辑 V3 ---
                if neighbor_type == 'synthetic':
                    # 如果匹配到合成缺陷，分数 = 相似度 (0.8 ~ 1.0)
                    # 越像缺陷，分越高
                    score = similarity
                else:
                    # 如果匹配到正常样本，分数 = 距离 (0.0 ~ 0.5)
                    # 越像正常，分越低
                    score = 1.0 - similarity
                
                k_scores.append(score)
            
            # 取平均 (Smooth out noise)
            avg_score = np.mean(k_scores)
            
            # 叠加
            anomaly_map[y:y+self.input_size, x:x+self.input_size] += avg_score
            count_map[y:y+self.input_size, x:x+self.input_size] += 1
            
        count_map[count_map == 0] = 1
        anomaly_map /= count_map
        
        # 3. 高斯平滑
        anomaly_map = gaussian_filter(anomaly_map, sigma=6) # 稍微加大平滑力度
        
        return anomaly_map, feats_np

    def apply_adaptive_threshold(self, heatmap):
        """核心改进：Otsu 自适应阈值去噪"""
        # 归一化到 0-255
        hm_min, hm_max = heatmap.min(), heatmap.max()
        if hm_max == hm_min: return heatmap # 避免除零
        
        hm_norm = (heatmap - hm_min) / (hm_max - hm_min)
        hm_uint8 = (hm_norm * 255).astype(np.uint8)
        
        # 使用 Otsu 算法自动寻找最佳阈值
        # Otsu 会寻找一个阈值，最大化类间方差（背景 vs 前景）
        otsu_thresh, _ = cv2.threshold(hm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # --- 关键策略 ---
        # Otsu 有时对只有背景的图也会切出一半作为前景。
        # 我们需要一个保护机制：如果全图分数都很低，说明全是好图，强制归零。
        if hm_max < 0.25: # 绝对安全阈值
            return np.zeros_like(heatmap)
            
        # 将低于 Otsu 阈值的区域强制归零 (背景抑制)
        # 将阈值转回 0-1 范围
        thresh_val = otsu_thresh / 255.0
        
        # 软截断：低于阈值的缓慢衰减，高于阈值的保留
        # 这里使用硬截断测试效果，如果太生硬可以改回软截断
        cleaned_map = heatmap.copy()
        cleaned_map[hm_norm < thresh_val] = 0
        
        return cleaned_map

    def visualize(self, img_path, ref_path, heatmap, save_name):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
        
        if ref_path and os.path.exists(ref_path):
            ref = cv2.imread(ref_path)
            ref = cv2.resize(ref, (512, 512))
        else:
            ref = np.zeros_like(img)
            
        # --- 使用自适应阈值处理热力图 ---
        heatmap = self.apply_adaptive_threshold(heatmap)
        # 调整热力图到与显示尺寸一致，避免布尔掩码与图像尺寸不匹配
        heatmap = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # 重新归一化以便可视化
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        hm_uint8 = np.uint8(255 * hm_norm)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        
        # 如果热力图是全0 (完美好图)，不要叠加蓝色，直接显示原图
        if heatmap.max() == 0:
            overlay = img
        else:
            # 仅在非零区域叠加颜色
            mask = hm_norm > 0.05
            blended = cv2.addWeighted(img, 0.6, hm_color, 0.4, 0)
            overlay = img.copy()
            overlay[mask] = blended[mask]
        
        concat = np.hstack((img, ref, overlay))
        cv2.imwrite(save_name, concat)

    def run_test(self):
        test_root = os.path.join(DATASET_ROOT, self.category, "test")
        save_dir = os.path.join(OUTPUT_DIR, self.category)
        os.makedirs(save_dir, exist_ok=True)
        
        subdirs = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
        
        for dtype in subdirs:
            img_dir = os.path.join(test_root, dtype)
            files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))][:3]
            
            for f in tqdm(files, desc=f"Infering {dtype}"):
                img_path = os.path.join(img_dir, f)
                pil_img = Image.open(img_path).convert("RGB")
                
                heatmap, patch_feats = self.compute_anomaly_map(pil_img)
                if heatmap is None: continue
                
                # 找参照
                global_feat_approx = np.mean(patch_feats, axis=0, keepdims=True)
                global_feat_approx /= np.linalg.norm(global_feat_approx)
                ref_path = self.get_reference_image_path(global_feat_approx)
                
                save_name = os.path.join(save_dir, f"{dtype}_{f}")
                self.visualize(img_path, ref_path, heatmap, save_name)

if __name__ == "__main__":
    engine = RAGInference(category="bottle")
    engine.run_test()
    print(f"\n✅ Inference Complete! Results saved to {OUTPUT_DIR}")
