"""
# 文件说明（data_tools/数据清洗.py）

- **文件作用**：将嵌套结构的行业知识 JSON 清洗并扁平化，生成可检索的文本知识库列表。
- **运行方式**：在项目根目录执行 `python Scripts/data_tools/数据清洗.py`，或在其他脚本中导入并调用 `build_text_knowledge_base`。
- **输出结果**：生成 `knowledge_corpus.json`，每条记录包含类别、键名、描述文本与类型标记。
- **分类角色**：归属于 `data_tools` 分类，是为 RAG 与大模型提示提供文本知识库的清洗脚本。
"""

import json
from pathlib import Path

def build_text_knowledge_base(json_path: str, output_path: str) -> None:
    """
    作用：把嵌套结构的 MVTec 行业知识（类别->缺陷/正常->描述）清洗并扁平化，生成可检索列表
    小白解释：原来是一个大字典，里面按类别和缺陷分层；我把它拆成一条一条的记录，
            每条记录包含类别、键名（good 或具体缺陷）、文本内容，以及是“正常标准”还是“缺陷定义”。
    参数：
      - json_path：输入 JSON 的路径（domain_knowledge.json）
      - output_path：输出 JSON 的保存路径（knowledge_corpus.json）
    返回：无，直接在 output_path 写出清洗后的列表
    """

    # 读取原始 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 取出 MVTec 总节点（如不存在，则用空字典）
    dataset_content = raw_data.get("MVTec", {})

    knowledge_entries = []

    # 遍历所有类别及其缺陷/正常描述
    for category, defects_map in dataset_content.items():
        if not isinstance(defects_map, dict):
            # 非预期结构，跳过该类别
            continue
        for defect_type, description_raw in defects_map.items():
            # 1) 文本清洗：去换行、首尾空格；确保是字符串
            clean_desc = str(description_raw).replace("\n", " ").strip()

            # 2) 标记类型：good 视为正常标准，其余视为缺陷定义
            is_normal = (defect_type == "good")

            entry = {
                "category": category,
                "key": defect_type,
                "content": clean_desc,
                "type": "normal_criteria" if is_normal else "defect_definition"
            }
            knowledge_entries.append(entry)

    # 写出为扁平化列表
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_entries, f, ensure_ascii=False, indent=4)

    # 统计信息输出
    categories = list(dataset_content.keys()) if isinstance(dataset_content, dict) else []
    normal_cnt = sum(1 for e in knowledge_entries if e["type"] == "normal_criteria")
    defect_cnt = sum(1 for e in knowledge_entries if e["type"] == "defect_definition")
    print(f"✅ 文本知识库构建完成！共处理 {len(knowledge_entries)} 条知识。")
    print(f"   - 包含类别: {categories}")
    print(f"   - 正常标准 normal_criteria: {normal_cnt}")
    print(f"   - 缺陷定义 defect_definition: {defect_cnt}")


if __name__ == "__main__":
    # 默认把输入/输出定位到与 domain_knowledge.json 同目录（DataSet/MVTec-AD 下）
    base = Path(r"d:\桌面\MRID\DataSet\MVTec-AD")
    inp = base / "domain_knowledge.json"
    out = base / "knowledge_corpus.json"
    build_text_knowledge_base(str(inp), str(out))
