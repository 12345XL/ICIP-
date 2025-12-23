# `dino` 目录说明

## 分类目的

`dino/` 目录集中放置基于 **DINOv2** 的视觉检索与缺陷热力图生成相关脚本，用来：

- 构建以 DINOv2 特征为基础的图像向量库（FAISS 索引）
- 基于向量库对测试集做检索，生成 `[原图 | 参照图 | 热力图]` 拼图
- 为后续的多模态大模型（如 Qwen3-VL）提供稳定的视觉提示（Visual Prompt）

## 文件清单

- `feature_extractor.py`  
  DINOv2 特征提取工具类，封装模型加载、预处理与特征归一化。

- `build_visual_bank_dino.py`  
  遍历 MVTec-AD 训练集正常图片，提取 DINOv2 特征并构建全局索引与 Patch 索引。

- `inject_negatives_dino.py`  
  在 DINOv2 的 Patch 索引中注入 CutPaste 伪缺陷 Patch，并为元数据补充 `type='synthetic'` 标记，强化 `inference_dino.py` 的异常敏感度。

- `inference_dino.py`  
  使用 DINOv2 向量库，对测试集图片生成缺陷热力图，并与原图、参照图拼接成可视化结果。

## 使用指南

典型使用顺序如下（以项目根目录为当前路径）：

1. **构建 DINO 向量库**

   ```bash
   cd /data/XL/多模态RAG
   python Scripts/dino/build_visual_bank_dino.py
   ```

   - 依赖：已解压好的 MVTec-AD 数据集，路径与脚本中 `DATASET_ROOT` 保持一致
   - 输出：`./embeddings_dino/` 目录，包含各类别的 `_global.index`、`_patch.index` 与 `*_meta.pkl`

2. **可选：为 DINO 索引注入伪造缺陷（负样本增强）**

   ```bash
   cd /data/XL/多模态RAG
   python Scripts/dino/inject_negatives_dino.py
   ```

   - 依赖：上一步生成的 `embeddings_dino/`
   - 输出：在各类别的 Patch 索引中追加一批 `type='synthetic'` Patch，并更新对应 `*_meta.pkl`

3. **生成 DINO 热力图与拼图**

   ```bash
   cd /data/XL/多模态RAG
   export TARGET_CATEGORY=bottle  # 或其他类别名称
   python Scripts/dino/inference_dino.py
   ```

   - 依赖：`embeddings_dino/`（如执行了上一步，会同时包含 normal 与 synthetic Patch）
   - 输出：`./results_dino_final/<category>/dtype_filename.png`
     - 左：测试原图
     - 中：检索到的正常参照图（Golden Reference）
     - 右：DINOv2 生成的缺陷热力图叠加图

3. **下游使用**

   - `Scripts/experiments/benchmark_qwen.py` 会读取 `results_dino_final/` 中的拼图，作为多模态大模型的视觉输入。

## 注意事项

- 建议在 **GPU 环境** 下运行，DINOv2 特征提取对算力要求较高。
- 如果服务器无法联网，需要提前将 DINOv2 权重下载到本地，并修改 `FeatureExtractor` 中的加载方式（例如改为 `torch.jit.load` 本地文件）。
- 默认使用固定的绝对路径：
  - 数据集根：`/data/XL/多模态RAG/DataSet/MVTec-AD`
  - 如需迁移到其他机器，请统一调整脚本中的路径常量。
