"""用于 IMU 人体活动识别的轻量模型。"""

from .mlp import NUM_CLASSES, TinyMLP, count_trainable_parameters

__all__ = ["NUM_CLASSES", "TinyMLP", "count_trainable_parameters"]
