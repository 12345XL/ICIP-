"""
# 文件说明（experiments/compare_models_oneshot.py）

- **文件作用**：在单个缺陷 Patch 上，对比 CLIP 与 DINOv2 的相似度表现，直观感受两者对细小缺陷的敏感度差异。
- **运行方式**：在项目根目录执行 `python Scripts/experiments/compare_models_oneshot.py`，可按需修改图片路径和裁剪坐标。
- **输出结果**：在终端打印两种模型的相似度分数，并生成 `comparison_visual.png` 方便肉眼查看 Patch 差异。
- **分类角色**：归属于 `experiments` 分类，是用于分析与展示模型特性的快速对比脚本。
"""

import torch
import clip
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- 配置 ---
# 1. 你的测试图 (有缺陷)
TEST_IMG_PATH = "/data/XL/多模态RAG/DataSet/MVTec-AD/bottle/test/broken_large/000.png" 
# 2. 找一张正常图做对比 (随便找一张 train/good 里的)
NORMAL_IMG_PATH = "/data/XL/多模态RAG/DataSet/MVTec-AD/bottle/train/good/000.png" 

# 3. 关注的坐标 (缺陷位置)
CROP_X, CROP_Y = 200, 200  
PATCH_SIZE = 224 # Patch 大小

class ModelComparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Device: {self.device}")
        
        # --- 加载 CLIP ---
        print("🚀 Loading CLIP (ViT-L/14@336px)...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_model.eval()

        # --- 加载 DINOv2 ---
        print("🦕 Loading DINOv2 (ViT-L/14)...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.dino_model.to(self.device)
        self.dino_model.eval()
        
        # DINOv2 预处理
        self.dino_preprocess = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_feature(self, img_pil, model_name):
        """提取特征并归一化"""
        with torch.no_grad():
            if model_name == 'clip':
                tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
                feat = self.clip_model.encode_image(tensor)
            else:
                tensor = self.dino_preprocess(img_pil).unsqueeze(0).to(self.device)
                feat = self.dino_model(tensor)
            
            # L2 归一化 (关键!)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]

    def run_comparison(self):
        if not os.path.exists(TEST_IMG_PATH) or not os.path.exists(NORMAL_IMG_PATH):
            print("❌ Image path error! Please check paths.")
            return

        # 1. 读取并裁剪 Patch
        img_bad = Image.open(TEST_IMG_PATH).convert("RGB")
        img_good = Image.open(NORMAL_IMG_PATH).convert("RGB")
        
        box = (CROP_X, CROP_Y, CROP_X + PATCH_SIZE, CROP_Y + PATCH_SIZE)
        patch_bad = img_bad.crop(box)
        patch_good = img_good.crop(box)

        # 2. CLIP 对比
        clip_feat_bad = self.get_feature(patch_bad, 'clip')
        clip_feat_good = self.get_feature(patch_good, 'clip')
        # 计算余弦相似度 (点积)
        clip_score = np.dot(clip_feat_bad, clip_feat_good)

        # 3. DINOv2 对比
        dino_feat_bad = self.get_feature(patch_bad, 'dino')
        dino_feat_good = self.get_feature(patch_good, 'dino')
        dino_score = np.dot(dino_feat_bad, dino_feat_good)

        # 4. 打印结果
        print("\n" + "="*40)
        print(f"📊 Model Comparison Report")
        print("="*40)
        print(f"Defect Patch vs Normal Patch Similarity:")
        print(f"🔹 CLIP Score:   {clip_score:.4f}  (Higher = More Similar)")
        print(f"🔸 DINOv2 Score: {dino_score:.4f}")
        print("-" * 40)
        
        if clip_score > 0.9 and dino_score < 0.8:
            print("✅ 结论: DINOv2 成功发现了差异，而 CLIP '瞎了'。")
            print("   (DINOv2 分数低说明它看出了两者不同，这是好事！)")
        elif clip_score > 0.9 and dino_score > 0.9:
            print("⚠️ 结论: 两个模型都没看出区别。可能 Patch 位置没切准，或缺陷太不明显。")
        else:
            print("ℹ️ 结论: 观察分数差距。通常 DINOv2 的分数应该显著低于 CLIP。")

        # 5. 可视化确认
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(patch_bad)
        plt.title(f"Defect Patch\n(Test Image)")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(patch_good)
        plt.title(f"Normal Reference\n(Train Image)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("comparison_visual.png")
        print("🖼️ Patches saved to 'comparison_visual.png'. Check if they look different.")

if __name__ == "__main__":
    comp = ModelComparator()
    comp.run_comparison()
