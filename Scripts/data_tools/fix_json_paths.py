"""
# 文件说明（data_tools/fix_json_paths.py）

- **文件作用**：批量修正各类 JSON 文件中的旧路径字符串，统一替换为当前机器上的绝对路径。
- **运行方式**：在项目根目录执行 `python Scripts/data_tools/fix_json_paths.py`，内部默认遍历 `TARGET_ROOT`。
- **输出结果**：就地覆盖原 JSON 文件，控制台打印被修复的文件列表与统计信息。
- **分类角色**：归属于 `data_tools` 分类，是数据集清洗与路径规范化的小工具脚本。
"""

import os
import json
import re
from typing import Any

BASE_DIR = "/data/XL/多模态RAG/DataSet/MVTec-AD"
TARGET_ROOT = "/data/XL/多模态RAG/DataSet"

def _rewrite_single_path(path_str: str) -> str:
    """
    函数作用：把 JSON 里的旧路径字符串改成系统里的真实绝对路径。
    小白解释：有些 JSON 里写的是类似 "../../../dataset/MVTec/bottle/..." 的老路径，
    我们把里面的 "dataset/MVTec/" 识别出来，替换成你机器上的 "MVTec-AD" 真实位置。
    """
    if not isinstance(path_str, str):
        return path_str
    s = path_str.replace("\\", "/")
    key = "dataset/MVTec/"
    if key in s:
        idx = s.find(key)
        rest = s[idx + len(key):]
        # 拼接为绝对路径
        new_path = os.path.join(BASE_DIR, rest)
        new_path = new_path.replace("\\", "/")
        return new_path
    # 如果是已经是绝对路径但前缀错误，也做一次规范化
    if s.startswith("/data/XL/多模态RAG/DataSet/MVTec/"):
        rest = s.split("/DataSet/MVTec/")[-1]
        new_path = os.path.join(BASE_DIR, rest)
        return new_path.replace("\\", "/")
    return path_str

def _process_obj(obj: Any) -> Any:
    """
    函数作用：递归遍历 JSON 结构，遇到字符串路径就尝试修复；遇到列表/字典就继续深入。
    小白解释：不管 JSON 多复杂，我们一层层往里走，凡是路径字符串就改成正确的绝对路径。
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # 特殊处理 normal_captions_index.json 里的 similar_paths: [[score, path], ...]
            if k == "similar_paths" and isinstance(v, list):
                new_list = []
                for item in v:
                    if isinstance(item, list) and len(item) == 2:
                        score, p = item
                        new_list.append([score, _rewrite_single_path(p)])
                    else:
                        new_list.append(_process_obj(item))
                new_obj[k] = new_list
            elif k in ("image_paths", "path", "file", "filepath") and isinstance(v, str):
                new_obj[k] = _rewrite_single_path(v)
            else:
                new_obj[k] = _process_obj(v)
        return new_obj
    elif isinstance(obj, list):
        return [_process_obj(x) for x in obj]
    elif isinstance(obj, str):
        return _rewrite_single_path(obj)
    else:
        return obj

def fix_json_paths(root_dir: str) -> int:
    """
    函数作用：遍历指定目录下的所有 .json 文件，统一修复其中的路径。
    小白解释：从给定的文件夹开始，把里面每个 JSON 文件都打开、修改路径、再保存。
    返回值：成功修改的文件数量。
    """
    changed = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    data = json.load(f)
                new_data = _process_obj(data)
                # 只有在内容发生变化时写回
                if json.dumps(data, ensure_ascii=False, sort_keys=True) != json.dumps(new_data, ensure_ascii=False, sort_keys=True):
                    with open(full, "w", encoding="utf-8") as f:
                        json.dump(new_data, f, ensure_ascii=False, indent=4)
                    changed += 1
                    print(f"修复: {full}")
            except Exception as e:
                print(f"跳过(解析失败或权限问题): {full} -> {e}")
    return changed

def _quick_verify_sample():
    """
    函数作用：快速核对几个典型 JSON 是否已修复到绝对路径（示例验证）。
    小白解释：随便抽几份 JSON 看一眼，确认路径已经是以 "/data/XL/多模态RAG/DataSet/MVTec-AD/" 开头。
    """
    samples = [
        os.path.join(BASE_DIR, "bottle", "normal_captions_index.json"),
        os.path.join(BASE_DIR, "capsule", "normal_captions_index.json"),
        os.path.join(BASE_DIR, "carpet", "normal_captions_index.json"),
    ]
    for fp in samples:
        if not os.path.exists(fp):
            print(f"示例不存在: {fp}")
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # 找到第一条记录，打印 image_paths
            for k, v in obj.items():
                if isinstance(v, dict) and "image_paths" in v:
                    p = v["image_paths"]
                    print("示例路径:", p[:120])
                    break
        except Exception as e:
            print(f"示例读取失败: {fp} -> {e}")

if __name__ == "__main__":
    total = fix_json_paths(TARGET_ROOT)
    print(f"总计修复 JSON 文件: {total}")
    _quick_verify_sample()
