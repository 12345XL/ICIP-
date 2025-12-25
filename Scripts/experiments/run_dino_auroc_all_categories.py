import os
import json
import subprocess
from typing import List, Optional, Tuple


SCRIPTS_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)


def run_cmd(cmd: List[str], env: Optional[dict] = None) -> None:
    """小白解释：这个函数帮你在命令行里自动执行一条指令，就像你自己在终端里敲命令一样；如果失败会抛错，方便你发现问题。"""
    print(f"\n[Run] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败：{' '.join(cmd)}，返回码={result.returncode}")


def list_mvtec_categories(dataset_root: str) -> List[str]:
    """小白解释：这个函数会去 MVTec 数据集根目录，把所有类别文件夹名（bottle、cable 等）都列出来，后面就可以自动循环处理。"""
    cats: List[str] = []
    for name in os.listdir(dataset_root):
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path):
            cats.append(name)
    cats.sort()
    return cats


def find_latest_run_dir(tag: str, category: str) -> Optional[str]:
    """小白解释：DINO 推理每次都会在 Results/exp 里新建一个 run 目录，这个函数帮你在所有 run 中找到“某个类别+某个 tag 对应的最新一次 run”。"""
    exp_root = os.path.join(PROJECT_ROOT, "Results", "exp")
    if not os.path.exists(exp_root):
        return None
    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(exp_root):
        run_dir = os.path.join(exp_root, name)
        if not os.path.isdir(run_dir):
            continue
        cfg_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        if cfg.get("tag") != tag:
            continue
        if cfg.get("category") != category:
            continue
        mtime = os.path.getmtime(run_dir)
        candidates.append((mtime, run_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def main() -> None:


    """小白解释：这是只负责“DINO 推理 + AUROC 评估”的一键脚本，不会跑 Qwen；它会遍历所有类别，对每个类别生成一个 run 并计算 metrics.json。"""
    import sys
    sys.path.append(SCRIPTS_DIR)
    from config import DATASET_ROOT

    py_bin = os.environ.get("PYTHON_BIN", "/data/XL/Myidea/bin/python")
    index_id = os.environ.get("INDEX_ID", "mvtec_dinov2_default")

    print(f"使用 Python 解释器: {py_bin}")
    print(f"使用 INDEX_ID: {index_id}")

    categories = list_mvtec_categories(DATASET_ROOT)
    print(f"检测到 MVTec-AD 类别: {categories}")

    env_base = os.environ.copy()
    env_base["INDEX_ID"] = index_id

    print("\n========== 仅运行 DINO 推理 + AUROC（不包含 Qwen） ==========")
    for cat in categories:
        tag = f"dino_baseline_{cat}"
        print(f"\n------ 处理类别: {cat}，run tag: {tag} ------")

        env = env_base.copy()
        env["TARGET_CATEGORY"] = cat
        env["RUN_TAG"] = tag

        print(f"\n[阶段 1] DINO 推理生成热力图和 panels（类别：{cat}）")
        run_cmd([py_bin, "Scripts/dino/inference_dino.py"], env=env)

        run_dir = find_latest_run_dir(tag, cat)
        if not run_dir:
            raise RuntimeError(f"找不到类别 {cat} 且 tag={tag} 对应的 run_dir，请检查 DINO 推理是否成功。")
        print(f"[Info] 使用 run_dir: {run_dir}")

        print(f"\n[阶段 2] 评估 AUROC（类别：{cat}）")
        run_cmd(
            [py_bin, "Scripts/experiments/eval_anomaly_maps.py", "--run_dir", run_dir],
            env=env_base,
        )

    print("\n✅ 所有类别的 DINO 推理 + AUROC 评估已完成，请在 Results/exp 下查看各自 run 的 scores/metrics.json。")


if __name__ == "__main__":
    main()
