import os
import json
from typing import List, Dict, Any

import pandas as pd


SCRIPTS_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)


def collect_run_metrics(exp_root: str) -> List[Dict[str, Any]]:
    """小白解释：这个函数会遍历 Results/exp 下面所有的 run 目录，把每个 run 里的 AUROC 指标读出来，整理成一个列表。"""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(exp_root):
        return rows
    for name in os.listdir(exp_root):
        run_dir = os.path.join(exp_root, name)
        if not os.path.isdir(run_dir):
            continue
        cfg_path = os.path.join(run_dir, "run_config.json")
        metrics_path = os.path.join(run_dir, "scores", "metrics.json")
        if not (os.path.exists(cfg_path) and os.path.exists(metrics_path)):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            continue
        category = cfg.get("category") or metrics.get("category")
        tag = cfg.get("tag", "")
        image_auc = metrics.get("image_level_AUROC", None)
        pixel_auc = metrics.get("pixel_level_AUROC", None)
        rows.append(
            {
                "run_id": name,
                "tag": tag,
                "category": category,
                "image_level_AUROC": image_auc,
                "pixel_level_AUROC": pixel_auc,
            }
        )
    return rows


def main() -> None:
    """小白解释：这是一个小汇总脚本，会把所有 run 的 AUROC 指标收集起来，存成一个 CSV 表，方便你一眼看出每个类别大概表现。"""
    exp_root = os.path.join(PROJECT_ROOT, "Results", "exp")
    rows = collect_run_metrics(exp_root)
    if not rows:
        print("⚠️ 没有找到任何 metrics.json，请先运行 DINO 推理和 AUROC 评估。")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["category", "run_id"])
    out_path = os.path.join(PROJECT_ROOT, "Results", "summary_dino_auroc.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ 已汇总 {len(df)} 条 run 记录到 {out_path}")


if __name__ == "__main__":
    main()

