"""
# 文件说明（dino/inference_dino.py）

- **文件作用**：基于 DINOv2 向量库，对测试集图片生成 `[原图|参照图|热力图]` 拼图，用于后续 RAG + 大模型评测。
- **运行方式**：在项目根目录执行 `python Scripts/dino/inference_dino.py`，可通过环境变量 `TARGET_CATEGORY` 指定类别。
- **输出结果**：在 `results_dino_final/<category>/` 下保存拼接好的可视化图片。
- **分类角色**：归属于 `dino` 分类，是 DINO 视觉检索管线中的「推理与可视化」脚本。
"""

import os
import sys
import torch
import faiss
import pickle
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

SCRIPTS_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.append(SCRIPTS_DIR)
sys.path.append(PROJECT_ROOT)
from feature_extractor import FeatureExtractor
from config import DATASET_ROOT
from utils.paths import (
    make_run_dir,
    save_run_config,
    make_index_id,
    index_root,
    category_dir,
)

STRIDE_RATIO = float(os.environ.get("STRIDE_RATIO", "0.5"))
HEATMAP_SIGMA = float(os.environ.get("HEATMAP_SIGMA", "4"))
SCORE_MODE = os.environ.get("SCORE_MODE", "1nn").strip().lower()
TRIM_M = int(os.environ.get("TRIM_M", "2"))
TOPK = 5

class DINOInference:
    def __init__(self, category, run_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.category = category
        self.run_dir = run_dir
        self.extractor = FeatureExtractor(device=self.device)
        self.input_size = self.extractor.input_size
        self.stride_ratio = STRIDE_RATIO
        self.stride = int(self.input_size * self.stride_ratio)
        self.topk = TOPK
        self.heatmap_sigma = HEATMAP_SIGMA
        self.score_mode = SCORE_MODE
        self.trim_m = TRIM_M
        backbone = "dinov2"
        env_index_id = os.environ.get("INDEX_ID")
        if env_index_id:
            self.index_id = env_index_id
        else:
            self.index_id = make_index_id(
                backbone=backbone,
                input_size=self.input_size,
                stride=self.stride,
                topk=self.topk,
                extra="normal",
            )
        idx_root = index_root(self.index_id)
        cat_idx_dir = category_dir(idx_root, category)
        patch_index_path = os.path.join(cat_idx_dir, "patch.index")
        meta_path = os.path.join(cat_idx_dir, "meta.pkl")
        global_index_path = os.path.join(cat_idx_dir, "global.index")
        print(f"📂 Loading DINO indices for {category} from {cat_idx_dir} ...")
        if not os.path.exists(patch_index_path):
            raise FileNotFoundError(f"Index not found: {patch_index_path}. Did you run build_visual_bank_dino.py?")
        self.patch_index = faiss.read_index(patch_index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
        self.patch_info = self.meta["patch_info"]
        types = [p.get("type", "normal") for p in self.patch_info]
        if any(t != "normal" for t in types):
            raise RuntimeError(
                "patch.index 里检测到非 normal patch（可能混入了 synthetic）。"
                "请重新运行 build_visual_bank_dino.py 重建 normal-only index，或清理后再推理。"
            )
        if os.path.exists(global_index_path):
            self.global_index = faiss.read_index(global_index_path)
            self.global_paths = self.meta.get("global_paths", [])
        else:
            self.global_index = None
            self.global_paths = []

    def get_reference_image_path(self, img_feat):
        """全局检索参照图"""
        if self.global_index is None: return None
        # DINO 特征维度是 1024
        D, I = self.global_index.search(img_feat, 1)
        ref_idx = I[0][0]
        if ref_idx < len(self.global_paths):
            return self.global_paths[ref_idx]
        return None

    def compute_anomaly_map(self, img_pil):
        """计算热力图（one-class 距离法，离正常越远分数越高）"""
        w, h = img_pil.size
        patches = []
        coords = []
        
        ys = list(range(0, h - self.input_size + 1, self.stride))
        xs = list(range(0, w - self.input_size + 1, self.stride))
        if ys and ys[-1] != h - self.input_size:
            ys.append(h - self.input_size)
        if xs and xs[-1] != w - self.input_size:
            xs.append(w - self.input_size)

        for y in ys:
            for x in xs:
                box = (x, y, x + self.input_size, y + self.input_size)
                patches.append(self.extractor.preprocess(img_pil.crop(box)))
                coords.append((x, y))
        
        if not patches: return None, None

        feats_np = self.extractor.encode(patches)

        K = self.topk
        D, I = self.patch_index.search(feats_np, K)
        
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        for i, (x, y) in enumerate(coords):
            if self.score_mode == "1nn":
                patch_score = float(D[i, 0])
            elif self.score_mode == "topk_mean":
                patch_score = float(np.mean(D[i, :K]))
            elif self.score_mode == "trimmed_mean":
                m = min(max(1, self.trim_m), K)
                patch_score = float(np.mean(np.sort(D[i, :K])[:m]))
            else:
                raise ValueError(f"Unknown SCORE_MODE={self.score_mode}")
            anomaly_map[y:y+self.input_size, x:x+self.input_size] += patch_score
            count_map[y:y+self.input_size, x:x+self.input_size] += 1
            
        count_map[count_map == 0] = 1
        anomaly_map /= count_map
        
        # 高斯平滑（可控）：sigma<=0 时不平滑
        if self.heatmap_sigma > 0:
            anomaly_map = gaussian_filter(anomaly_map, sigma=self.heatmap_sigma)
        
        return anomaly_map, feats_np

    def apply_adaptive_threshold(self, heatmap):
        """
        修改版：移除暴力截断，保留微弱信号，并打印调试信息
        """
        hm_min, hm_max = heatmap.min(), heatmap.max()
        
        # 1. 打印调试信息 (关键！)
        # 如果这个值很小(比如0.05)，说明DINO觉得两张图一模一样
        print(f"   [Debug] Heatmap Peak Score: {hm_max:.4f}") 
        
        if hm_max == hm_min: 
            return np.zeros_like(heatmap)
        
        # 2. 暂时注释掉这个“绝对安全阈值”
        # 之前的逻辑是：如果最高分都没超过0.15，就当做全黑。
        # 现在我们把它关掉，看看是不是微小缺陷得分只有 0.1 左右
        # if hm_max < 0.15: 
        #     return np.zeros_like(heatmap)
            
        # 归一化到 0-1
        hm_norm = (heatmap - hm_min) / (hm_max - hm_min)
        hm_uint8 = (hm_norm * 255).astype(np.uint8)
        
        # 3. 使用 Otsu 阈值 (这是自适应的，比较安全)
        otsu_thresh, _ = cv2.threshold(hm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = otsu_thresh / 255.0
        
        # 4. 放宽 Otsu：只切掉 Otsu 阈值的一半，保留更多红色
        # 这样即使信号微弱，也会显示出来
        cleaned_map = heatmap.copy()
        cleaned_map[hm_norm < (thresh_val * 0.5)] = 0 
        
        return cleaned_map
    def visualize(self, img_path, ref_path, heatmap, save_name):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
        
        if ref_path and os.path.exists(ref_path):
            ref = cv2.imread(ref_path)
            ref = cv2.resize(ref, (512, 512))
        else:
            ref = np.zeros_like(img)
            
        # 去噪
        heatmap = self.apply_adaptive_threshold(heatmap)
        # 调整热力图到显示尺寸，避免掩码与图像尺寸不一致
        heatmap = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # 归一化可视化
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        hm_uint8 = np.uint8(255 * hm_norm)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        
        if heatmap.max() == 0:
            overlay = img
        else:
            mask = hm_norm > 0.05
            blended = cv2.addWeighted(img, 0.6, hm_color, 0.4, 0)
            overlay = img.copy()
            overlay[mask] = blended[mask]
        
        concat = np.hstack((img, ref, overlay))
        cv2.imwrite(save_name, concat)

    def run_test(self):
        test_root = os.path.join(DATASET_ROOT, self.category, "test")
        raw_base = os.path.join(self.run_dir, "inference", "raw_heatmaps")
        pan_base = os.path.join(self.run_dir, "inference", "panels")
        scores_path = os.path.join(self.run_dir, "scores", "scores_image.csv")
        raw_cat = category_dir(raw_base, self.category)
        pan_cat = category_dir(pan_base, self.category)
        records = []
        if not os.path.exists(test_root):
            return
        subdirs = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
        print(f"Detecting defect types for {self.category}: {subdirs}")
        for dtype in subdirs:
            img_dir = os.path.join(test_root, dtype)
            files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
            for f in tqdm(files, desc=f"Infering {self.category}/{dtype}"):
                img_path = os.path.join(img_dir, f)
                pil_img = Image.open(img_path).convert("RGB")
                heatmap, _ = self.compute_anomaly_map(pil_img)
                if heatmap is None:
                    continue
                raw_heatmap = heatmap.astype(np.float32)
                img_score = float(np.max(raw_heatmap))
                gt_label = 0 if dtype == "good" else 1
                g_feat = self.extractor.encode([self.extractor.preprocess(pil_img)])
                ref_path = self.get_reference_image_path(g_feat)
                stem = f"{dtype}_{os.path.splitext(f)[0]}"
                raw_path = os.path.join(raw_cat, f"{stem}.npy")
                np.save(raw_path, raw_heatmap)
                panel_path = os.path.join(pan_cat, f"{stem}.png")
                self.visualize(img_path, ref_path, raw_heatmap, panel_path)
                records.append(
                    {
                        "category": self.category,
                        "dtype": dtype,
                        "filename": f"{dtype}/{f}",
                        "gt_label": gt_label,
                        "img_score": img_score,
                        "heatmap_path": raw_path,
                        "vis_path": panel_path,
                    }
                )
        if records:
            import pandas as pd
            df = pd.DataFrame(records)
            df.to_csv(scores_path, index=False)
            print(f"✅ Scores saved to {scores_path}")


if __name__ == "__main__":
    target = os.environ.get("TARGET_CATEGORY", "bottle")
    tag = os.environ.get("RUN_TAG", f"dino_{target}")
    run_dir = make_run_dir(tag)
    engine = DINOInference(category=target, run_dir=run_dir)
    cfg = {
        "tag": tag,
        "category": target,
        "index_id": engine.index_id,
        "backbone": "dinov2",
        "input_size": engine.input_size,
        "stride": engine.stride,
        "topk": engine.topk,
        "stride_ratio": engine.stride_ratio,
        "score_mode": engine.score_mode,
        "heatmap_sigma": engine.heatmap_sigma,
        "dataset": "MVTec-AD",
    }
    save_run_config(run_dir, cfg)
    engine.run_test()
    print(f"\n✅ DINO Inference Complete for {target}! Check results in {run_dir}")
