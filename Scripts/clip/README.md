# `clip` 目录说明

## 分类目的

`clip/` 目录集中放置基于 **CLIP (ViT-L/14@336px)** 的向量库构建、负样本注入和可视化推理脚本，用来：

- 快速验证 RAG 思路在 CLIP 特征上的表现
- 构建 CLIP 版本的全局 / Patch 向量索引
- 通过热力图与可视化手段分析 CLIP 对缺陷的敏感度

## 文件清单

- `build_visual_bank.py`  
  使用 CLIP 对训练集正常图片提取全局与 Patch 特征，构建 `embeddings/` 向量库。

- `inject_negatives.py`  
  基于 CutPaste 方法，为 CLIP 向量库注入伪造缺陷 Patch，扩充 `type='synthetic'` 的负样本。

- `inject_negatives_dino.py`  
  功能与上一个脚本类似，但增加了向量维度一致性检查，避免索引与特征维度不匹配。

- `inference.py`  
  使用 CLIP 向量库，对测试集生成带热力图叠加的拼图结果（用于直观观察 CLIP 的检索效果）。

- `debug.py`  
  显微镜式调试工具：查看某个 Patch 在向量库中的 Top-K 结果，并以图片形式展示邻居 Patch。

## 使用指南

一个典型的 CLIP 管线流程如下：

1. **构建 CLIP 向量库**

   ```bash
   cd /data/XL/多模态RAG
   python Scripts/clip/build_visual_bank.py
   ```

   - 输出：`./embeddings/` 目录，包括 `<category>_global.index`、`<category>_patch.index` 与 `*_meta.pkl`

2. **可选：注入伪造缺陷（负样本增强）**

   ```bash
   # 普通版本
   python Scripts/clip/inject_negatives.py

   # 带维度校验的增强版本
   python Scripts/clip/inject_negatives_dino.py
   ```

   - 输出：在原有 `embeddings/` 索引基础上追加若干 `type='synthetic'` Patch

3. **生成 CLIP 热力图可视化**

   ```bash
   python Scripts/clip/inference.py
   ```

   - 输出：`./results_visualization_v3/<category>/dtype_filename.png`
   - 用于对比 CLIP 和 DINOv2 热力图质量的直观效果

4. **显微镜调试模式**

   ```bash
   python Scripts/clip/debug.py
   ```

   - 输出：`debug_report.png`，展示查询 Patch 及其 Top-K 邻居 Patch

## 注意事项

- CLIP 权重通过 `clip.load("ViT-L/14@336px")` 下载，如在离线环境使用，需要提前缓存或改为本地加载。
- 所有脚本默认假设数据集位于 `/data/XL/多模态RAG/DataSet/MVTec-AD`，如路径不同需统一修改常量。
- 与 DINO 管线相比，CLIP 更偏向对语义的理解，对细粒度纹理缺陷的敏感度可能稍弱，可通过本目录脚本进行对比验证。

