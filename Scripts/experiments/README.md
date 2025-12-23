# `experiments` 目录说明

## 分类目的

`experiments/` 目录存放所有 **实验性与分析类脚本**，主要用于：

- 评测多模态 RAG 流水线的整体效果（例如 Qwen3-VL + DINO RAG）
- 对比不同视觉编码器（CLIP vs DINOv2）的表现
- 进行快速原型验证与可视化实验

## 文件清单

- `benchmark_qwen.py`  
  使用 Qwen3-VL-4B-Instruct，对比 **无 RAG** 与 **DINO RAG + 行业知识** 两种模式下的缺陷检测指标。

- `compare_models_oneshot.py`  
  在单个缺陷 Patch 上比较 CLIP 与 DINOv2 的相似度分数，并输出对比可视化图像。

## 使用指南

1. **运行 Qwen 多模态 RAG 评测**

   前置条件：

   - 已完成：
     - DINO 向量库构建：`Scripts/dino/build_visual_bank_dino.py`
     - DINO 推理生成热力图：`Scripts/dino/inference_dino.py`
     - 行业知识清洗：`Scripts/data_tools/数据清洗.py`（生成 `knowledge_corpus.json`）
   - 已正确配置本地 Qwen3-VL 模型路径 `MODEL_PATH`

   评测命令：

   ```bash
   cd /data/XL/多模态RAG
   python Scripts/experiments/benchmark_qwen.py
   ```

   - 输出：
     - `benchmark_results_<CATEGORY>.csv`
     - 终端打印 No-RAG 与 With-RAG 的 Accuracy / Precision / Recall / F1

2. **对比 CLIP 与 DINOv2 在单个 Patch 上的表现**

   ```bash
   cd /data/XL/多模态RAG
   python Scripts/experiments/compare_models_oneshot.py
   ```

   - 可在脚本顶部修改：
     - `TEST_IMG_PATH`（带缺陷的测试图）
     - `NORMAL_IMG_PATH`（正常参考图）
     - `CROP_X, CROP_Y, PATCH_SIZE`（关注的局部区域）
   - 输出：`comparison_visual.png`，便于肉眼观察两个 Patch 的差异，同时在终端打印两种模型的余弦相似度。

## 注意事项

- 本目录脚本主要面向 **分析与实验** 场景，不建议直接用于生产推理流程。
- 多模态评测依赖项较多（数据集、向量库、热力图、知识库、VLM 模型等），运行前建议按 README 中的顺序逐步准备。

