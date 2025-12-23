import os
import os.path as osp

PROJECT_ROOT = osp.dirname(osp.dirname(__file__))


def get_dataset_root() -> str:
    """
    小白解释：优先从环境变量 MVTEC_ROOT 里读数据集根路径，
    如果你没设置，就用项目里默认的本地路径。
    """
    return os.environ.get("MVTEC_ROOT", "/data/XL/多模态RAG/DataSet/MVTec-AD")


def get_embed_dino_dir() -> str:
    """
    小白解释：控制 DINO 特征库（FAISS 索引）的保存目录，
    默认是项目根目录下的 embeddings_dino 文件夹。
    """
    default_dir = osp.join(PROJECT_ROOT, "embeddings_dino")
    return os.environ.get("EMBED_DINO_DIR", default_dir)


def get_out_dino_dir() -> str:
    """
    小白解释：控制 DINO 推理结果（拼图、热力图等）的输出目录，
    默认是项目根目录下的 results_dino_final 文件夹。
    """
    default_dir = osp.join(PROJECT_ROOT, "results_dino_final")
    return os.environ.get("OUT_DINO_DIR", default_dir)


def get_qwen_model_path() -> str:
    """
    小白解释：指定本地 Qwen3-VL 模型的权重路径，
    如果你挪了模型位置，可以设置环境变量 QWEN_MODEL_PATH 来覆盖。
    """
    default_path = (
        "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
        "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
    )
    return os.environ.get("QWEN_MODEL_PATH", default_path)


def get_knowledge_path() -> str:
    """
    小白解释：行业知识库 JSON 的路径，默认放在数据集根目录下，
    也可以用环境变量 KNOWLEDGE_PATH 单独指定。
    """
    default_path = osp.join(get_dataset_root(), "knowledge_corpus.json")
    return os.environ.get("KNOWLEDGE_PATH", default_path)


# 兼容老代码的常量写法（方便直接 from config import XXX）
DATASET_ROOT = get_dataset_root()
EMBED_DINO_DIR = get_embed_dino_dir()
OUT_DINO_DIR = get_out_dino_dir()
QWEN_MODEL_PATH = get_qwen_model_path()
KNOWLEDGE_PATH = get_knowledge_path()
