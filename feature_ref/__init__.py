"""IMU TinyML HAR 项目的 Python 特征参考实现。

包顶层只暴露日常调用需要的稳定接口。典型用法是把同一窗口的 9 个通道组织成字典，
调用 ``extract_features(signals)`` 得到 39 维 float32 数组，再使用仅由训练集拟合的
标准化参数处理后送入 MLP。调试某一维时，可通过 ``FEATURE_NAMES`` 查看完整顺序，或用
``FEATURE_INDEX[特征名]`` 查询从 0 开始的索引。

底层数学函数仍可从 ``feature_ref.time_features`` 显式导入，但它们不在包顶层重复导出，
以保持主要接口简洁。
"""

# 聚合常用提取函数和 schema 常量，使调用方无需了解内部文件划分。
from .extractor import extract_feature_dict, extract_features
from .feature_schema import FEATURE_DIM, FEATURE_INDEX, FEATURE_NAMES, WINDOW_SIZE

# __all__ 同时声明受支持的公共 API，并控制 ``from feature_ref import *`` 的结果。
__all__ = [
    "FEATURE_DIM",
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "WINDOW_SIZE",
    "extract_feature_dict",
    "extract_features",
]
