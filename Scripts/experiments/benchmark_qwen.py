"""
# 文件说明（experiments/benchmark_qwen.py）

- **文件作用**：对比「直接问大模型」和「DINO RAG + 热力图 + 行业知识」两种模式下的缺陷检测效果。
- **运行方式**：在项目根目录执行 `python Scripts/experiments/benchmark_qwen.py`，确保已生成 `results_dino_final` 和 `knowledge_corpus.json`。
- **输出结果**：生成 `benchmark_results_<CATEGORY>.csv`，并在终端打印 Accuracy / Precision / Recall / F1 对比报告。
- **分类角色**：归属于 `experiments` 分类，是整条多模态 RAG 流水线的综合评测脚本。
"""

import os
import sys
import json
import argparse
import torch
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SCRIPTS_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.append(SCRIPTS_DIR)
sys.path.append(PROJECT_ROOT)
from config import (
    DATASET_ROOT,
    QWEN_MODEL_PATH,
    KNOWLEDGE_PATH,
)

MODEL_PATH = QWEN_MODEL_PATH

class IndustrialBenchmark:
    def __init__(self, category, panel_dir, out_dir):
        """初始化基准评测器：加载 Qwen3-VL 模型与处理器，并准备知识库。
        小白解释：这一步就是把多模态大模型（能看图会说话）从本地路径加载进来，
        同时加载一个专门的处理器（Processor）用来把图片和文字变成模型能理解的格式。
        另外还把你写好的行业知识库读进来，后面作为提示词的一部分使用。"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.category = category
        self.panel_dir = panel_dir
        self.out_dir = out_dir
        print(f"🚀 Loading Qwen Model from: {MODEL_PATH}")
        
        # 加载模型 (自动适配显存)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        # 加载知识库 (用于 RAG 模式)
        self.knowledge_text = self._load_knowledge()
        print("✅ Model & Knowledge Loaded.")

    def _binary_from_text(self, text):
        s = (text or "").lower()
        head = s[:20]
        if ("yes" in head) or ("fail" in head) or ("defect" in head):
            return 1
        return 0

    def _load_knowledge(self):
        """读取并整理当前类别的行业知识，生成可插入提示词的文本。
        小白解释：把 JSON 里的知识筛选出来，拼成一段说明文字，
        模型看到这段说明就更懂行业标准和缺陷定义。"""
        with open(KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        # 筛选当前类别的知识
        normal_desc = ""
        defects = []
        for item in kb:
            if item['category'] == self.category:
                if item['type'] == 'normal_criteria':
                    normal_desc += item['content']
                else:
                    defects.append(f"- {item['key']}: {item['content']}")
        
        return f"**Normal Standards:**\n{normal_desc}\n\n**Potential Defects:**\n" + "\n".join(defects)

    def predict(self, image_path, prompt_text):
        """通用推理：给模型一张图和一段文字，生成回答文本。
        小白解释：这里先把你的问题和图片打包成对话格式，
        然后用处理器把它们变成张量，最后让模型生成答案并解码成字符串。"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # 预处理
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

    def run_experiment(self):
        """批量运行评测：遍历测试集，对无 RAG 与有 RAG 两种设置分别推理并记录结果。
        小白解释：循环跑测试集的图片，先直接问模型（无知识），
        再用我们生成的拼图+知识（有 RAG）去问一次，对比结果并存表格。"""
        test_root = os.path.join(DATASET_ROOT, self.category, "test")
        results = []
        
        # 遍历所有测试子文件夹
        subdirs = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
        
        print(f"🔥 Starting Benchmark on {self.category}...")
        
        for dtype in subdirs:
            # 标记 Ground Truth (good=0, 缺陷=1)
            label = 0 if dtype == "good" else 1
            img_dir = os.path.join(test_root, dtype)
            
            # 为了快速出结果，每个类型只跑 10 张图 (你可以改成所有)
            files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))][:10]
            
            for f in tqdm(files, desc=f"Testing {dtype}"):
                raw_img_path = os.path.join(img_dir, f)
                
                # --- Setting A: No RAG (Baseline) ---
                # 只有原图，没有背景知识，直接问
                prompt_a = "Look at this product image. Is there any defect? Answer strictly with 'Yes' or 'No' first, then explain."
                pred_a_text = self.predict(raw_img_path, prompt_a)
                pred_a_score = self._binary_from_text(pred_a_text)
                
                # --- Setting B: With RAG (Ours) ---
                # 使用 inference_dino.py 生成的组合图 (Visual Prompt)
                # 组合图路径 (需确保你之前跑过 inference_dino.py)
                stem = f"{dtype}_{os.path.splitext(f)[0]}"
                rag_img_path = os.path.join(self.panel_dir, self.category, f"{stem}.png")
                
                if os.path.exists(rag_img_path):
                    prompt_b = f"""
ROLE: You are a Quality Assurance AI analyzing an industrial product.

INPUT LAYOUT:
- Left Panel: Original Product Image.
- Middle Panel: Golden Reference (Perfect Standard).
- Right Panel: Anomaly Score Map (heatmap).

HEATMAP DESCRIPTION:
1. The Right Panel is an anomaly score map produced by an algorithm.
2. Warmer colors (yellow/red) indicate higher probability of defect, cooler colors (blue) indicate lower probability.
3. The heatmap may contain noise or imperfect signals. Do not blindly trust every red pixel.
4. You must jointly consider the original image, the reference image and the heatmap before making a decision.

Domain Knowledge for {self.category}:
{self.knowledge_text}

TASK:
1. Decide whether this product is defective (answer Yes or No).
2. Provide a confidence score between 0 and 1 (0 = definitely normal, 1 = definitely defective).
3. Briefly describe the main suspicious region if any (location and type).

RESPONSE FORMAT (ENGLISH):
Line 1: "Answer: Yes" or "Answer: No"
Line 2: "Confidence: <number between 0 and 1>"
Line 3: One short sentence describing the key evidence.
"""
                    pred_b_text = self.predict(rag_img_path, prompt_b)
                    pred_b_score = self._binary_from_text(pred_b_text)
                else:
                    # 如果没有对应的 RAG 图片（之前没生成），则跳过或设为 -1
                    pred_b_score = -1 
                    pred_b_text = "RAG Image Missing"

                # 记录结果
                results.append({
                    "filename": f"{dtype}/{f}",
                    "gt_label": label,
                    "pred_no_rag": pred_a_score,
                    "pred_with_rag": pred_b_score,
                    "expl_no_rag": pred_a_text,
                    "expl_with_rag": pred_b_text
                })

        # 保存表格
            df = pd.DataFrame(results)
            os.makedirs(self.out_dir, exist_ok=True)
            pred_csv = os.path.join(self.out_dir, "predictions.csv")
            df.to_csv(pred_csv, index=False)
            print(f"\n💾 Raw results saved to {pred_csv}")
            return df

    def calculate_metrics(self, df):
        """计算硬性指标：Accuracy、Precision、Recall、F1。
        小白解释：把预测结果和真值做对比，算出常见的分类指标，
        并打印一个对比报告，方便你看 RAG 是否提升效果。"""
        # 过滤掉 RAG 图片缺失的数据
        df_valid = df[df["pred_with_rag"] != -1]
        
        y_true = df_valid["gt_label"].values
        y_pred_a = df_valid["pred_no_rag"].values
        y_pred_b = df_valid["pred_with_rag"].values
        
        def get_metrics(y_t, y_p):
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            return {
                "Accuracy": accuracy_score(y_t, y_p),
                "Precision": precision_score(y_t, y_p, zero_division=0),
                "Recall": recall_score(y_t, y_p, zero_division=0),
                "F1": f1_score(y_t, y_p, zero_division=0)
            }
            
        metrics_a = get_metrics(y_true, y_pred_a)
        metrics_b = get_metrics(y_true, y_pred_b)
        
        print("\n" + "="*40)
        print("🏆 Final Benchmark Report")
        print("="*40)
        print(f"{'Metric':<12} | {'No RAG (Baseline)':<18} | {'With RAG (Ours)':<18}")
        print("-" * 52)
        for k in metrics_a.keys():
            print(f"{k:<12} | {metrics_a[k]:.4f}{' '*12} | {metrics_b[k]:.4f}")
        print("="*40)
        metrics = {
            "Accuracy_no_rag": metrics_a["Accuracy"],
            "Precision_no_rag": metrics_a["Precision"],
            "Recall_no_rag": metrics_a["Recall"],
            "F1_no_rag": metrics_a["F1"],
            "Accuracy_with_rag": metrics_b["Accuracy"],
            "Precision_with_rag": metrics_b["Precision"],
            "Recall_with_rag": metrics_b["Recall"],
            "F1_with_rag": metrics_b["F1"],
        }
        os.makedirs(self.out_dir, exist_ok=True)
        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"💾 Metrics saved to {metrics_path}")


def main():
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
    panel_dir = os.path.join(run_dir, "inference", "panels")
    out_dir = os.path.join(run_dir, "benchmark_qwen")
    benchmark = IndustrialBenchmark(category=category, panel_dir=panel_dir, out_dir=out_dir)
    df_results = benchmark.run_experiment()
    benchmark.calculate_metrics(df_results)


if __name__ == "__main__":
    main()
