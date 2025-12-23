# `data_tools` 目录说明

## 分类目的

`data_tools/` 目录包含所有与 **数据清洗与路径修复** 相关的小工具脚本，用来：

- 统一修复 JSON 文件中的路径字段，确保与当前机器上的真实目录一致
- 把原始行业知识 JSON 扁平化为可检索的文本知识库，供 RAG 与大模型使用

## 文件清单

- `fix_json_paths.py`  
  递归遍历指定根目录下的所有 `.json` 文件，修正其中的旧路径字符串，改写为绝对路径。

- `数据清洗.py`  
  将嵌套结构的行业知识 JSON（例如 `domain_knowledge.json`）清洗为扁平的 `knowledge_corpus.json` 列表。

## 使用指南

1. **修复 JSON 内的路径**

   ```bash
   cd /data/XL/多模态RAG
   python Scripts/data_tools/fix_json_paths.py
   ```

   - 默认遍历：`/data/XL/多模态RAG/DataSet`
   - 典型修复内容：
     - `"../../../dataset/MVTec/bottle/..."` → `"/data/XL/多模态RAG/DataSet/MVTec-AD/bottle/..."`
   - 控制台会打印已修复的文件路径与总数

2. **构建文本知识库**

   ```bash
   cd /data/XL/多模态RAG
   python Scripts/data_tools/数据清洗.py
   ```

   > 提示：脚本中默认使用 Windows 本地路径示例，实际项目中建议在调用时手动传入 `json_path` 和 `output_path`，或按你的目录结构进行修改。

   - 输出：`knowledge_corpus.json`
   - 内容：每条记录包含 `category`、`key`、`content` 与 `type` 字段，方便在 RAG 或 Prompt 中拼接使用。

## 注意事项

- `fix_json_paths.py` 会 **就地覆盖** 原 JSON 文件，建议在首次运行前做好备份。
- 如果迁移到新的服务器或更改了数据集目录结构，建议优先运行路径修复脚本，以免下游脚本读取不到数据。
- `数据清洗.py` 中的默认路径目前指向本地 Windows 示例，迁移到 Linux 路径时记得统一调整。

