import os
import json
from datetime import datetime


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


# 项目根目录：..（utils 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def make_run_id(tag: str = "baseline") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = tag.replace(" ", "_")
    return f"{ts}_{tag}"


def results_root() -> str:
    """
    所有结果统一放在 项目根目录/Results 下，与当前工作目录无关。
    """
    return ensure_dir(os.path.join(PROJECT_ROOT, "Results"))


def make_run_dir(tag: str = "baseline") -> str:
    root = results_root()
    run_id = make_run_id(tag)
    run_dir = ensure_dir(os.path.join(root, "exp", run_id))

    # 统一创建子目录（预留 evidence / faithfulness）
    for sub in [
        "logs",
        "inference/raw_heatmaps",
        "inference/overlays",
        "inference/panels",
        "inference/evidence",
        "faithfulness",
        "scores",
        "benchmark_qwen/failures/fp",
        "benchmark_qwen/failures/fn",
    ]:
        ensure_dir(os.path.join(run_dir, sub))

    return run_dir


def save_run_config(run_dir: str, cfg: dict) -> None:
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def make_index_id(
    backbone: str,
    input_size: int,
    stride: int,
    topk: int,
    extra: str = "normal",
) -> str:
    # 命名可按需调整，尽量短但包含关键信息
    return f"mvtec_{backbone}_sz{input_size}_s{stride}_k{topk}_{extra}"


def index_root(index_id: str) -> str:
    root = results_root()
    return ensure_dir(os.path.join(root, "indexes", index_id))


def category_dir(base: str, category: str) -> str:
    # 自动创建按类别的子目录
    return ensure_dir(os.path.join(base, category))