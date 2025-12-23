"""
# 文件说明（experiments/eval_anomaly_maps.py）

- **文件作用**：对 DINO 生成的热力图进行严肃评估，计算图像级 AUROC 和像素级 AUROC。
- **运行方式**：在项目根目录执行 `python Scripts/experiments/eval_anomaly_maps.py`，确保已跑完 `dino/inference_dino.py`。
- **输出结果**：在 `results_dino_final/<category>/metrics.json` 中保存评估指标。
"""

import os
import sys
import json
from typing import Optional, List, Tuple

import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPTS_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.append(SCRIPTS_DIR)
sys.path.append(PROJECT_ROOT)
from config import DATASET_ROOT


def evaluate_image_level(df: pd.DataFrame) -> float:
    """
    小白解释：使用每张图片的 `img_score` 和真值标签 `gt_label`，
    计算图像级别的 AUROC（越接近 1 说明区分正常/异常越好）。
    """
    y_true = df["gt_label"].values.astype(int)
    y_score = df["img_score"].values.astype(float)
    return float(roc_auc_score(y_true, y_score))


def collect_pixel_pairs(
    category: str, df: pd.DataFrame
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    小白解释：遍历所有有缺陷的样本，为每张图加载：
    1）算法输出的热力图 raw（float），
    2）MVTec 提供的 mask（二值图，1 表示缺陷像素），
    然后把所有图像的像素打平，凑成两个长向量用于计算像素级 AUROC。
    """
    y_true_list: List[np.ndarray] = []
    y_score_list: List[np.ndarray] = []

    for _, row in df.iterrows():
        if int(row["gt_label"]) == 0:
            continue

        filename = str(row["filename"])
        heatmap_path = str(row["heatmap_path"])

        if not os.path.exists(heatmap_path):
            continue

        dtype, img_name = filename.split("/", 1)
        base_name = os.path.splitext(os.path.basename(img_name))[0]

        mask_path = os.path.join(
            DATASET_ROOT,
            category,
            "ground_truth",
            dtype,
            f"{base_name}_mask.png",
        )
        if not os.path.exists(mask_path):
            continue

        heatmap = np.load(heatmap_path).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        if mask.shape != heatmap.shape:
            mask = cv2.resize(
                mask,
                (heatmap.shape[1], heatmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_bin = (mask > 0).astype(np.uint8)

        y_true_list.append(mask_bin.flatten())
        y_score_list.append(heatmap.flatten())

    if not y_true_list:
        return None

    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate(y_score_list, axis=0)
    return y_true, y_score


def evaluate_pixel_level(category: str, df: pd.DataFrame) -> Optional[float]:
    """
    小白解释：如果能找到对应的 mask，就计算像素级 AUROC，
    否则返回 None 表示当前类别没有像素级评估结果。
    """
    pairs = collect_pixel_pairs(category, df)
    if pairs is None:
        return None
    y_true, y_score = pairs
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to a run directory under Results/exp/",
    )
    args = parser.parse_args()
    run_dir = args.run_dir

    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    category = cfg.get("category")
    if not category:
        raise ValueError("category is missing in run_config.json")

    scores_path = os.path.join(run_dir, "scores", "scores_image.csv")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"scores_image.csv not found: {scores_path}")

    print(f"🔍 Evaluating anomaly maps for category: {category}")
    df = pd.read_csv(scores_path)
    img_auc = evaluate_image_level(df)
    px_auc = evaluate_pixel_level(category, df)

    print(f"Image-level AUROC: {img_auc:.4f}")
    if px_auc is not None:
        print(f"Pixel-level AUROC: {px_auc:.4f}")
    else:
        print("Pixel-level AUROC: N/A (no valid masks found)")

    metrics = {
        "category": category,
        "image_level_AUROC": img_auc,
        "pixel_level_AUROC": px_auc,
    }
    out_path = os.path.join(run_dir, "scores", "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✅ Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
