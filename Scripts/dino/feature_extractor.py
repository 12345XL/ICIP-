"""
# 文件说明（dino/feature_extractor.py）

- **文件作用**：封装 DINOv2 特征提取逻辑，把输入图片统一转换成可检索的向量。
- **运行方式**：作为工具类被其他脚本导入使用，例如 `build_visual_bank_dino.py`、`inference_dino.py`。
- **输出结果**：对外提供 `preprocess` 和 `encode` 方法，返回归一化的特征张量或 Numpy 向量。
- **分类角色**：归属于 `dino` 分类，是整条 DINO 视觉检索与推理管线的底层工具。
"""

import os
import torch
import torchvision.transforms as T


class FeatureExtractor:
    """DINOv2 特征提取器
    小白解释：这个类帮你把图片变成 DINOv2 的特征向量。
    先把图片预处理成模型喜欢的大小和数值范围，再用 DINOv2 提取特征，
    并做一次 L2 归一化（让向量长度统一，便于相似度比较）。"""

    def __init__(self, device: str | None = None):
        """初始化 DINOv2 模型与预处理管道
        小白解释：从官方仓库加载 ViT-L/14 的 DINOv2 模型，
        设置到 GPU 或 CPU 上，并准备一个把图片缩放到 224x224、标准化的预处理器。"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        local_repo = os.environ.get(
            "DINOV2_LOCAL_REPO", "/data/torchhub/hub/facebookresearch_dinov2_main"
        )
        try:
            self.model = torch.hub.load(
                local_repo, "dinov2_vitl14", source="local"
            )
        except Exception as e:
            raise RuntimeError(
                "无法从本地缓存加载 DINOv2，请检查 DINOV2_LOCAL_REPO 路径是否可用。错误：" + str(e)
            )
        self.model.to(self.device)
        self.model.eval()

        # 输入尺寸（ViT-L/14 用 224）
        self.input_size = 224

        # 预处理：Resize 到 224，转成张量并按 ImageNet 统计值做标准化
        self._preprocess = T.Compose([
            T.Resize((self.input_size, self.input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, pil_img):
        """把 PIL 图片预处理成模型输入张量
        小白解释：把图片缩放到 224，并做标准化，返回一个 `C×H×W` 的张量。"""
        return self._preprocess(pil_img)

    def encode(self, inputs):
        """批量提取特征并 L2 归一化，返回 numpy.float32 数组
        小白解释：把一批图片张量堆成一个批次，送进模型，得到特征后统一做长度归一化，
        最后转成 Numpy 数组给后续 FAISS 检索用。"""
        if isinstance(inputs, (list, tuple)):
            batch = torch.stack(inputs).to(self.device)
        else:
            batch = inputs.to(self.device)

        with torch.no_grad():
            feats = self.model(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype('float32')
