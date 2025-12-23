"""
# 文件说明（clip/debug.py）

- **文件作用**：显微镜式地查看某个 Patch 在 CLIP 向量库中的 Top-K 检索结果，帮助理解打分逻辑。
- **运行方式**：在项目根目录执行 `python Scripts/clip/debug.py`，可根据需要修改 `CATEGORY` 和 `DEBUG_X/DEBUG_Y`。
- **输出结果**：生成 `debug_report.png`，展示查询 Patch 及其邻居 Patch 的可视化对比。
- **分类角色**：归属于 `clip` 分类，是 CLIP 管线的「调试与可视化分析」脚本。
"""

import os
import clip
import torch
import faiss
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# --- 配置 ---
DATASET_ROOT = "/data/XL/多模态RAG/DataSet/MVTec-AD"
EMBEDDING_DIR = "./embeddings"
MODEL_NAME = "ViT-L/14@336px"
STRIDE_RATIO = 0.5 
CATEGORY = "bottle" 

TEST_IMG_PATH = os.path.join(DATASET_ROOT, CATEGORY, "test", "broken_large", "003.png")
if not os.path.exists(TEST_IMG_PATH):
    good_dir = os.path.join(DATASET_ROOT, CATEGORY, "test", "good")
    files = [f for f in os.listdir(good_dir) if f.lower().endswith((".png", ".jpg"))]
    TEST_IMG_PATH = os.path.join(good_dir, files[0]) if files else good_dir

DEBUG_X, DEBUG_Y = 200, 200 

class Debugger:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading CLIP...")
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)
        self.input_size = self.model.visual.input_resolution
        self.stride = int(self.input_size * STRIDE_RATIO)

        print(f"📂 Loading indices...")
        self.patch_index = faiss.read_index(os.path.join(EMBEDDING_DIR, f"{CATEGORY}_patch.index"))
        with open(os.path.join(EMBEDDING_DIR, f"{CATEGORY}_meta.pkl"), 'rb') as f:
            self.meta = pickle.load(f)
        self.patch_info = self.meta['patch_info']

    def inspect_patch(self, img_path, x, y):
        """显微镜模式：查看指定 Patch 的检索详情"""
        img_pil = Image.open(img_path).convert("RGB")
        
        # 1. 裁剪出关注的 Patch
        # 注意：这里需要对齐网格。我们找离 (x, y) 最近的那个切片网格点。
        grid_y = (y // self.stride) * self.stride
        grid_x = (x // self.stride) * self.stride
        
        box = (grid_x, grid_y, grid_x + self.input_size, grid_y + self.input_size)
        patch_pil = img_pil.crop(box)
        
        print(f"\n🔍 Inspecting Patch at grid ({grid_x}, {grid_y})...")
        
        # 2. 提特征
        input_tensor = self.preprocess(patch_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(input_tensor)
            feat /= feat.norm(dim=-1, keepdim=True)
            feat_np = feat.cpu().numpy().astype('float32')

        # 3. 检索 Top-5
        K = 5
        D, I = self.patch_index.search(feat_np, K)
        
        # 4. 可视化报告
        plt.figure(figsize=(15, 6))
        
        # 显示 Query Patch
        plt.subplot(1, K+1, 1)
        plt.imshow(patch_pil)
        plt.title(f"Query\n({grid_x},{grid_y})")
        plt.axis('off')
        
        print(f"{'Rank':<5} | {'Type':<10} | {'Score':<10} | {'Info'}")
        print("-" * 40)

        for k in range(K):
            idx = I[0][k]
            sim = D[0][k]
            info = self.patch_info[idx]
            p_type = info.get('type', 'normal')
            
            # 计算得分 (V3 逻辑)
            if p_type == 'synthetic':
                score = sim
                title_color = 'red'
            else:
                score = 1.0 - sim
                title_color = 'blue'
            
            print(f"{k+1:<5} | {p_type:<10} | {sim:.4f} -> {score:.4f} | {info.get('path', '')[-20:]}")

            # 尝试复原邻居 Patch (如果是 normal 类型，我们可以去原图里抠出来看)
            plt.subplot(1, K+1, k+2)
            
            if p_type == 'normal':
                parent_idx = info.get('parent_idx')
                src_path = None
                if parent_idx is not None:
                    gp = self.meta.get('global_paths', [])
                    if 0 <= parent_idx < len(gp):
                        src_path = gp[parent_idx]
                sx, sy = info['coords']
                if src_path and os.path.exists(src_path):
                    src_img = Image.open(src_path).convert("RGB")
                    neighbor_patch = src_img.crop((sx, sy, sx+self.input_size, sy+self.input_size))
                    plt.imshow(neighbor_patch)
                else:
                    plt.text(0.5, 0.5, "Img Lost", ha='center')
            else:
                blank = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
                blank[:, :, 0] = 255
                plt.imshow(blank)
                plt.text(self.input_size//2, self.input_size//2, "Synthetic\n(In-Memory)", ha='center', color='white')
            
            plt.title(f"{p_type}\nSim: {sim:.2f}", color=title_color, fontweight='bold')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig("debug_report.png")
        print("\n✅ Debug report saved to 'debug_report.png'. Open it to see details!")

if __name__ == "__main__":
    debugger = Debugger()
    debugger.inspect_patch(TEST_IMG_PATH, DEBUG_X, DEBUG_Y)
