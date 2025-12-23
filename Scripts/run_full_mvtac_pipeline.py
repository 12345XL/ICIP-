import os
import json
import subprocess
from typing import List, Optional, Tuple


SCRIPTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)


def run_cmd(cmd: List[str], env: Optional[dict] = None) -> None:
    """
    小白解释：这个小工具函数用来在命令行里帮你执行一条命令。
    就好像你手动在终端里敲命令一样，只是现在由代码来自动敲。
    如果命令执行失败，会直接抛出错误，方便你发现问题。
    """
    print(f"\n[Run] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=SCRIPTS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败：{' '.join(cmd)}，返回码={result.returncode}")


def list_mvtec_categories(dataset_root: str) -> List[str]:
    """
    小白解释：这个函数会去 MVTec-AD 数据集的根目录下面，
    把所有的类别文件夹名字（bottle、cable、capsule 等）都列出来，
    这样后面就可以自动对每个类别依次跑实验。
    """
    cats = []
    for name in os.listdir(dataset_root):
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path):
            cats.append(name)
    cats.sort()
    return cats


def find_latest_run_dir(tag: str, category: str) -> Optional[str]:
    """
    小白解释：DINO 推理脚本每次会创建一个新的 Results/exp/<run_id>/ 目录。
    这个函数会在所有 run 里面，找到 tag 和 category 都匹配、且时间最新的那个 run，
    这样我们就知道应该用哪一次的结果来做 AUROC 和 Qwen 评测。
    """
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
    """
    小白解释：这是整条流水线的一键脚本。
    它会做四件事：
    1）根据数据集自动找到所有类别；
    2）用同一个 INDEX_ID 构建或复用 DINO 索引；
    3）对每个类别跑一次 DINO 推理，生成一个 run（Results/exp/...）；
    4）对每个 run 先算 AUROC，再跑一次 Qwen benchmark，对比 No RAG 和 With RAG。
    你只要配置好 Python 路径、本地 DINOv2 模型、INDEX_ID，就可以一键跑全类。
    """
    from config import DATASET_ROOT

    py_bin = os.environ.get("PYTHON_BIN", "/data/XL/Myidea/bin/python")
    index_id = os.environ.get("INDEX_ID", "mvtec_dinov2_default")

    print(f"使用 Python 解释器: {py_bin}")
    print(f"使用 INDEX_ID: {index_id}")

    categories = list_mvtec_categories(DATASET_ROOT)
    print(f"检测到 MVTec-AD 类别: {categories}")

    env_base = os.environ.copy()
    env_base["INDEX_ID"] = index_id

    print("\n========== 第一步：构建或更新 DINO 索引 ==========")
    run_cmd([py_bin, "dino/build_visual_bank_dino.py"], env=env_base)

    print("\n========== 第二步：逐类跑 DINO 推理 + AUROC + Qwen ==========")
    for cat in categories:
        tag = f"dino_baseline_{cat}"
        print(f"\n------ 处理类别: {cat}，run tag: {tag} ------")

        env = env_base.copy()
        env["TARGET_CATEGORY"] = cat
        env["RUN_TAG"] = tag

        print(f"\n[阶段 2.1] DINO 推理生成热力图和 panels（类别：{cat}）")
        run_cmd([py_bin, "dino/inference_dino.py"], env=env)

        run_dir = find_latest_run_dir(tag, cat)
        if not run_dir:
            raise RuntimeError(f"找不到类别 {cat} 且 tag={tag} 对应的 run_dir，请检查 DINO 推理是否成功。")
        print(f"[Info] 使用 run_dir: {run_dir}")

        print(f"\n[阶段 2.2] 评估 AUROC（类别：{cat}）")
        run_cmd(
            [py_bin, "experiments/eval_anomaly_maps.py", "--run_dir", run_dir],
            env=env_base,
        )

        print(f"\n[阶段 2.3] Qwen 基准评测（类别：{cat}）")
        run_cmd(
            [py_bin, "experiments/benchmark_qwen.py", "--run_dir", run_dir],
            env=env_base,
        )

    print("\n✅ 全部类别的 DINO + AUROC + Qwen 基准评测已完成，请在 Results/exp 下查看每次 run 的详细结果。")


if __name__ == "__main__":
    main()

