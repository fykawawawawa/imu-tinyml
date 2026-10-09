"""面向 INT8/TFLM 部署的轻量 PyTorch MLP。

模型输入必须遵循 :mod:`feature_ref.feature_schema` 定义的 39 维固定顺序。
网络仅使用全连接层和 ReLU，避免引入端侧不必要的算子与激活内存。
输出为六类活动的 logits；训练时直接传给 ``CrossEntropyLoss``，推理时取
``argmax``，不要在模型内部添加 Softmax。
"""

from __future__ import annotations

from torch import Tensor, nn

from feature_ref.feature_schema import FEATURE_DIM

NUM_CLASSES = 6
HIDDEN_DIMS = (32, 16)


class TinyMLP(nn.Module):
    """使用 39 维窗口特征进行六分类的轻量 MLP。

    结构为 ``39 -> 32 -> 16 -> 6``，默认共 1,910 个可训练参数。模型返回
    未归一化 logits，供 PyTorch 训练及 TFLite/TFLM 转换使用。
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: tuple[int, int] = HIDDEN_DIMS,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        if input_dim != FEATURE_DIM:
            raise ValueError(
                f"input_dim must match feature schema ({FEATURE_DIM}), got {input_dim}"
            )
        if len(hidden_dims) != 2 or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain two positive integers")
        if num_classes != NUM_CLASSES:
            raise ValueError(f"num_classes must be {NUM_CLASSES}, got {num_classes}")

        hidden_1, hidden_2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """返回形状为 ``(..., 6)`` 的分类 logits。"""
        if features.ndim < 1 or features.shape[-1] != FEATURE_DIM:
            actual_shape = tuple(features.shape)
            raise ValueError(
                f"expected final dimension {FEATURE_DIM}, got shape {actual_shape}"
            )
        return self.network(features)


def count_trainable_parameters(model: nn.Module) -> int:
    """返回模型的可训练参数总数。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["HIDDEN_DIMS", "NUM_CLASSES", "TinyMLP", "count_trainable_parameters"]
