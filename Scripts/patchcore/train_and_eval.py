import os
import subprocess
import argparse
import sys

# 默认路径配置
DEFAULT_PYTHON_ENV = "/data/XL/pathcore/bin/python"
PATCHCORE_ROOT = "/data/XL/多模态RAG/Pathcore/patchcore-inspection"
DEFAULT_DATA_PATH = "/data/XL/多模态RAG/Pathcore/mvtec"
DEFAULT_RESULTS_PATH = "/data/XL/多模态RAG/Pathcore/patchcore-inspection/results_scripts"

def run_training(category, gpu_id, save_model=True):
    """
    封装 PatchCore 训练命令
    对应: bin/run_patchcore.py
    """
    print(f"[*] Starting PatchCore Training for category: {category}")
    
    # 构造命令参数
    cmd = [
        DEFAULT_PYTHON_ENV,
        "bin/run_patchcore.py",
        "--gpu", str(gpu_id),
        "--seed", "0",
        "--log_project", "MVTecAD_Results",
    ]
    
    if save_model:
        cmd.append("--save_patchcore_model")
        
    # 定义 Log Group 名称 (模型存放文件夹名)
    log_group = f"IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_{category.upper()}"
    cmd.extend(["--log_group", log_group])
    
    # 输出路径
    cmd.append(DEFAULT_RESULTS_PATH)
    
    # PatchCore 核心参数 (Backbone: WideResNet50)
    cmd.extend([
        "patch_core",
        "-b", "wideresnet50",
        "-le", "layer2",
        "-le", "layer3",
        "--pretrain_embed_dimension", "1024",
        "--target_embed_dimension", "1024",
        "--anomaly_scorer_num_nn", "1",
        "--patchsize", "3"
    ])
    
    # 注意: 去掉 --faiss_on_gpu 以兼容 CPU Faiss 环境
    
    # Sampler 参数 (Coreset 10%)
    cmd.extend([
        "sampler",
        "-p", "0.1",
        "approx_greedy_coreset"
    ])
    
    # Dataset 参数
    cmd.extend([
        "dataset",
        "--resize", "256",
        "--imagesize", "224",
        "-d", category,
        "mvtec",
        DEFAULT_DATA_PATH
    ])
    
    # 执行命令
    print(f"[*] Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" # 确保能找到 patchcore 包
    
    try:
        subprocess.run(cmd, cwd=PATCHCORE_ROOT, env=env, check=True)
        print(f"[+] Training finished successfully for {category}")
        return log_group
    except subprocess.CalledProcessError as e:
        print(f"[-] Training failed for {category} with error code {e.returncode}")
        return None

def run_evaluation(category, log_group, gpu_id, save_segmentation=True):
    """
    封装 PatchCore 评估命令
    对应: bin/load_and_evaluate_patchcore.py
    """
    print(f"[*] Starting PatchCore Evaluation for category: {category}")
    
    # 模型路径
    model_path = os.path.join(
        DEFAULT_RESULTS_PATH, 
        "MVTecAD_Results", 
        log_group, 
        "models", 
        f"mvtec_{category}"
    )
    
    if not os.path.exists(os.path.join(PATCHCORE_ROOT, model_path)):
        print(f"[-] Model path not found: {model_path}")
        return

    cmd = [
        DEFAULT_PYTHON_ENV,
        "bin/load_and_evaluate_patchcore.py",
        "--gpu", str(gpu_id),
        "--seed", "0",
    ]
    
    if save_segmentation:
        cmd.append("--save_segmentation_images")
        
    # 评估结果输出目录
    eval_results_dir = os.path.join(DEFAULT_RESULTS_PATH, "eval_results", log_group)
    cmd.append(eval_results_dir)
    
    # Loader 参数
    cmd.extend([
        "patch_core_loader",
        "-p", model_path
    ])
    
    # Dataset 参数 (评估集)
    cmd.extend([
        "dataset",
        "--resize", "256",
        "--imagesize", "224",
        "-d", category,
        "mvtec",
        DEFAULT_DATA_PATH
    ])
    
    print(f"[*] Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    try:
        subprocess.run(cmd, cwd=PATCHCORE_ROOT, env=env, check=True)
        print(f"[+] Evaluation finished. Results saved to: {eval_results_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[-] Evaluation failed with error code {e.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchCore 训练与评估适配器脚本")
    parser.add_argument("--category", type=str, required=True, help="MVTec 类别名 (例如: bottle)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，直接评估")
    parser.add_argument("--log_group", type=str, default=None, help="指定已有的模型 Log Group (仅评估时需要)")
    
    args = parser.parse_args()
    
    log_group = args.log_group
    
    if not args.skip_train:
        log_group = run_training(args.category, args.gpu)
    
    if log_group:
        run_evaluation(args.category, log_group, args.gpu)
    else:
        if args.skip_train:
            print("[-] Please provide --log_group when skipping training.")
