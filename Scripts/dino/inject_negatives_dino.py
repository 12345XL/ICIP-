"""
# 文件说明（dino/inject_negatives_dino.py）

- **文件作用**：在 DINOv2 的 Patch 向量库中注入 CutPaste 伪缺陷 Patch，并为原有 Patch 补充 `type` 标记，配合 `inference_dino.py` 的打分逻辑。
- **运行方式**：在项目根目录执行 `python Scripts/dino/inject_negatives_dino.py`，要求你已经先跑过 `build_visual_bank_dino.py` 构建好 `embeddings_dino/`。
- **输出结果**：在 `embeddings_dino/` 下的 `<category>_patch.index` 追加一批 `type='synthetic'` 的伪缺陷 Patch，并把对应元数据写入 `<category>_meta.pkl`。
- **分类角色**：归属于 `dino` 分类，是 DINOv2 管线的「负样本注入与难例强化」脚本，专门为 `inference_dino.py` 提供更“会思考”的索引库。
"""

import os
import sys
import faiss
import pickle
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from feature_extractor import FeatureExtractor
from config import DATASET_ROOT, EMBED_DINO_DIR

# --- 配置（必须与你 build_visual_bank_dino.py 保持一致） ---
EMBEDDING_DIR = EMBED_DINO_DIR  # DINOv2 索引目录
STRIDE_RATIO = 0.5
SYNTHETIC_RATIO = 0.5  # 每 2 张正常图，就大约生成 1 张“假缺陷”图量的 Patch


class DINONegativeInjector:
    """DINOv2 负样本注入器
    小白解释：这个类专门用来“伪造缺陷”，然后用 DINOv2 把这些伪缺陷编码成向量，
    最后追加到原有的 Patch 索引里，并在元数据里打上 `type='synthetic'` 的标签，
    方便 `inference_dino.py` 在打分时区分“像好人”还是“像坏人”。"""

    def __init__(self):
        """初始化 DINOv2 提取器与滑动窗口参数
        小白解释：这里会加载一次 DINOv2 模型（比较耗时），并根据模型的输入尺寸
        计算一个滑动步长，用来在图片上裁剪出一片片小 Patch 做特征提取。"""
        self.extractor = FeatureExtractor()
        self.input_size = self.extractor.input_size
        self.stride = int(self.input_size * STRIDE_RATIO)

    def generate_synthetic_defect(self, img_pil: Image.Image) -> Image.Image:
        """利用 CutPaste 思路在正常图上制造伪缺陷
        小白解释：简单理解，就是从图片上“抠”一小块，再做一点颜色/亮度增强，
        然后贴回到别的位置，让这张图看起来有一点“奇怪”的地方，模拟真实缺陷。"""
        w, h = img_pil.size

        # 1. 随机切一个小 Patch (约占整图 5%~15% 尺寸)
        patch_w = random.randint(int(w * 0.05), int(w * 0.15))
        patch_h = random.randint(int(h * 0.05), int(h * 0.15))

        src_x = random.randint(0, w - patch_w)
        src_y = random.randint(0, h - patch_h)
        patch = img_pil.crop((src_x, src_y, src_x + patch_w, src_y + patch_h))

        # 2. 对这块 Patch 做一系列“变形”，让它看起来更像坏点
        if random.random() > 0.5:
            patch = ImageOps.posterize(patch, 4)
        patch = ImageEnhance.Color(patch).enhance(random.uniform(0.5, 2.5))
        patch = ImageEnhance.Brightness(patch).enhance(random.uniform(0.6, 1.4))

        # 3. 把变形后的 Patch 贴回去（位置随机）
        dst_x = random.randint(0, w - patch_w)
        dst_y = random.randint(0, h - patch_h)
        aug_img = img_pil.copy()
        aug_img.paste(patch, (dst_x, dst_y))

        return aug_img

    def _extract_patches(
        self, img_pil: Image.Image
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """从一张图片中裁剪 Patch 并用 DINOv2 编码
        小白解释：这一步会在整张图片上“滑动窗口”，一格一格地裁出小块，
        对每个小块做预处理并送进 DINOv2 模型，最后得到一批 Patch 向量和对应坐标。"""
        w, h = img_pil.size
        batch_patches = []
        batch_coords: List[Tuple[int, int]] = []

        ys = list(range(0, h - self.input_size + 1, self.stride))
        xs = list(range(0, w - self.input_size + 1, self.stride))
        if ys and ys[-1] != h - self.input_size:
            ys.append(h - self.input_size)
        if xs and xs[-1] != w - self.input_size:
            xs.append(w - self.input_size)

        for y in ys:
            for x in xs:
                box = (x, y, x + self.input_size, y + self.input_size)
                patch = img_pil.crop(box)
                batch_patches.append(self.extractor.preprocess(patch))
                batch_coords.append((x, y))

        if not batch_patches:
            return np.zeros((0, self.input_size), dtype="float32"), []

        feats = self.extractor.encode(batch_patches)  # shape: [N, d]
        return feats, batch_coords

    def process_category(self, category: str) -> None:
        """针对某个大类，在其 DINO Patch 索引中注入伪缺陷
        小白解释：给定一个类别（例如 bottle），会：
        1）加载已有的 DINO Patch 索引和元数据；
        2）遍历 train/good 下的正常图，按一定比例挑一些图做 CutPaste 伪缺陷；
        3）用 DINOv2 编码这些伪缺陷 Patch，并写入索引和 meta。"""
        print(f"\n⚡ Injecting DINO Negatives for: {category}")

        patch_index_path = os.path.join(EMBEDDING_DIR, f"{category}_patch.index")
        meta_path = os.path.join(EMBEDDING_DIR, f"{category}_meta.pkl")

        if not os.path.exists(patch_index_path) or not os.path.exists(meta_path):
            print(f"   ❌ Skip {category}: patch index or meta not found.")
            return

        # 1. 加载现有索引和元数据
        index = faiss.read_index(patch_index_path)
        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)  # {'global_paths': [], 'patch_info': []}

        print(f"   -> Original Patch Count: {index.ntotal}")

        train_good_path = os.path.join(DATASET_ROOT, category, "train", "good")
        if not os.path.exists(train_good_path):
            print(f"   ❌ Train path not found: {train_good_path}")
            return

        img_files = [
            f
            for f in os.listdir(train_good_path)
            if f.lower().endswith((".png", ".jpg"))
        ]

        new_vectors = []
        new_patch_meta = []

        for img_idx, img_file in enumerate(
            tqdm(img_files, desc=f"Generating DINO Synthetics")
        ):
            # 按比例抽样，控制额外伪缺陷的数量
            if random.random() > SYNTHETIC_RATIO:
                continue

            img_path = os.path.join(train_good_path, img_file)
            try:
                pil_img = Image.open(img_path).convert("RGB")
                bad_img = self.generate_synthetic_defect(pil_img)

                feats, batch_coords = self._extract_patches(bad_img)
                if feats.shape[0] == 0:
                    continue

                # 收集向量
                new_vectors.append(feats)

                # 记录元数据，关键是 type='synthetic'
                for (x, y) in batch_coords:
                    new_patch_meta.append(
                        {
                            "parent_idx": -1,  # 伪缺陷不关联具体 global 图
                            "coords": (x, y),
                            "type": "synthetic",
                        }
                    )
            except Exception as e:
                print(f"   Error processing {img_path}: {e}")

        if not new_vectors:
            print("   ⚠️ No synthetic patches generated. Nothing to add.")
            return

        # 2. 追加并保存
        final_vectors = np.concatenate(new_vectors, axis=0).astype("float32")
        if final_vectors.shape[1] != index.d:
            raise ValueError(
                f"Embedding dim mismatch: vectors {final_vectors.shape[1]} vs index {index.d}. "
                f"请确认 DINOv2 模型与构库脚本 `build_visual_bank_dino.py` 使用的是同一个特征维度。"
            )

        index.add(final_vectors)

        # 给旧数据补上 type 标记（兼容早期版本）
        for item in meta_data.get("patch_info", []):
            if "type" not in item:
                item["type"] = "normal"

        # 追加新的 synthetic 元数据
        meta_data.setdefault("patch_info", [])
        meta_data["patch_info"].extend(new_patch_meta)

        faiss.write_index(index, patch_index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(meta_data, f)

        print(
            f"✅ Added {len(new_patch_meta)} synthetic DINO patches. New Size: {index.ntotal}"
        )


if __name__ == "__main__":
    """脚本入口
    小白解释：直接运行本文件，会遍历数据集中所有类别（如 bottle、capsule 等），
    对每个类别的 DINO Patch 索引注入一批伪造缺陷 Patch。
    建议在已经构建好 `embeddings_dino/` 之后再运行，且在首次大规模注入前备份一下索引目录。"""
    injector = DINONegativeInjector()
    categories = sorted(
        [
            d
            for d in os.listdir(DATASET_ROOT)
            if os.path.isdir(os.path.join(DATASET_ROOT, d))
        ]
    )
    for cat in categories:
        injector.process_category(cat)
