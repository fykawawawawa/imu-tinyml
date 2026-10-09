"""从 UCI HAR 惯性信号组合固定顺序的 39 维特征向量。

UCI HAR 的 ``Inertial Signals`` 目录将每个通道存成独立文本文件，同一行号表示同一个
2.56 秒窗口。本模块不直接读取文件，而是接收“通道名 -> 128 点数组”的映射，因此既可
用于数据集批处理，也可复用于实时采集到的单窗口数据。

输出顺序由 ``feature_schema.py`` 定义，并作为模型输入接口的一部分固定下来。提取过程
只包含时域统计、重力均值、向量模长统计和轴间相关性，不执行 FFT 或其他频域计算。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .feature_schema import (
    DYNAMIC_CHANNELS,
    FEATURE_DIM,
    FEATURE_NAMES,
    REQUIRED_CHANNELS,
    WINDOW_SIZE,
)
from .time_features import (
    as_float64_vector,
    axis_statistics,
    magnitude_statistics,
    safe_correlation,
    vector_magnitude,
)

# 对外输出类型：一维、长度为 FEATURE_DIM 的 float32 NumPy 数组。
FeatureVector = NDArray[np.float32]


def _validated_window(signals: Mapping[str, ArrayLike]) -> dict[str, NDArray[np.float64]]:
    """检查输入通道并构造内部计算窗口。

    必需的 9 个键由 ``REQUIRED_CHANNELS`` 定义。每个值必须是一维、恰好 128 点且全部
    有限。映射中的额外键会被忽略。

    返回的新字典只包含必需通道，值统一为 float64 数组。缺少键时抛出 KeyError；数据
    维度、长度或有限性不符合要求时抛出 ValueError。
    """
    # 一次性收集全部缺失通道，比遇到第一个键就失败更方便排查数据装载问题。
    missing = [name for name in REQUIRED_CHANNELS if name not in signals]
    if missing:
        raise KeyError(f"missing required channels: {', '.join(missing)}")

    window: dict[str, NDArray[np.float64]] = {}
    for name in REQUIRED_CHANNELS:
        # 先做通用的一维/有限性检查，再施加 UCI HAR 特有的 128 点约束。
        signal = as_float64_vector(signals[name], name=name)
        if signal.shape != (WINDOW_SIZE,):
            raise ValueError(
                f"{name} must contain exactly {WINDOW_SIZE} samples, "
                f"got shape {signal.shape}"
            )
        window[name] = signal
    return window


def extract_features(signals: Mapping[str, ArrayLike]) -> FeatureVector:
    """提取单个窗口固定顺序的 39 维特征向量。

    参数：
        signals: 通道名到数组状对象的映射。必需键为 ``body_acc_{x,y,z}``、
            ``body_gyro_{x,y,z}`` 和 ``total_acc_{x,y,z}``，每个通道必须恰好包含
            128 个有限值。

    返回：
        形状为 ``(39,)`` 的 float32 NumPy 数组。第 i 个值对应
        ``FEATURE_NAMES[i]``，也可通过 ``FEATURE_INDEX`` 查询具体索引。

    异常：
        KeyError: 缺少一个或多个必需通道。
        ValueError: 任一通道形状错误、长度不是 128，或包含 NaN/Inf。
        RuntimeError: 实际生成的特征数与固定 schema 不一致，通常表示代码改动造成漂移。

    数值约定：
        中间计算采用 float64，最后一次性转换为 float32。这样既保留 Python 参考实现的
        计算稳定性，并使输出直接匹配 MLP 输入及嵌入式实现的数据类型。
    """
    window = _validated_window(signals)
    features: list[float] = []

    # 1～24 维：6 个动态轴各自的 mean、std、abs_mean、peak_to_peak。
    # 不直接对 total_acc 做同样统计，避免它与 body_acc 和重力信息高度重复。
    for channel in DYNAMIC_CHANNELS:
        features.extend(axis_statistics(window[channel]))

    # 25～27 维：重力 = 总加速度 - 身体加速度，再对窗口求均值。
    # UCI HAR 已提供官方滤波后的 body_acc，因此无需在这里重新设计低通滤波器。
    for axis in "xyz":
        gravity = window[f"total_acc_{axis}"] - window[f"body_acc_{axis}"]
        features.append(float(np.mean(gravity)))

    # 28～33 维：身体加速度模长和角速度模长的均值、标准差、最大值。
    # 模长把 XYZ 合成一个旋转不变量，可补充单轴特征对安装方向较敏感的问题。
    acc_magnitude = vector_magnitude(
        window["body_acc_x"], window["body_acc_y"], window["body_acc_z"]
    )
    gyro_magnitude = vector_magnitude(
        window["body_gyro_x"], window["body_gyro_y"], window["body_gyro_z"]
    )
    features.extend(magnitude_statistics(acc_magnitude))
    features.extend(magnitude_statistics(gyro_magnitude))

    # 34～39 维：两组三轴信号各自的 xy、xz、yz 相关系数。
    # 相关性刻画多个轴是否同步或反向变化，不重复计算 yx、zx、zy。
    for group in ("body_acc", "body_gyro"):
        x = window[f"{group}_x"]
        y = window[f"{group}_y"]
        z = window[f"{group}_z"]
        features.extend(
            (
                safe_correlation(x, y),
                safe_correlation(x, z),
                safe_correlation(y, z),
            )
        )

    # 只在所有 float64 中间结果生成后统一降为 float32，避免逐项舍入累积差异。
    result = np.asarray(features, dtype=np.float32)
    # 防止修改计算逻辑后忘记同步固定 schema。
    if result.shape != (FEATURE_DIM,):
        raise RuntimeError(
            f"feature schema expects {FEATURE_DIM} values, produced {result.size}"
        )
    return result


def extract_feature_dict(signals: Mapping[str, ArrayLike]) -> dict[str, float]:
    """提取单个窗口，并返回“特征名 -> 特征值”字典。

    该接口便于调试、日志记录和人工查看；批量训练应优先使用 ``extract_features`` 返回的
    紧凑数组。``strict=True`` 保证名称数量与数值数量不一致时立即报错，而不是静默截断。
    字典保持 ``FEATURE_NAMES`` 的插入顺序，但模型输入仍应以数组 schema 为准。
    """
    values = extract_features(signals)
    return {name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=True)}


# 下划线开头的校验函数属于实现细节，不作为包的稳定接口导出。
__all__ = ["FeatureVector", "extract_feature_dict", "extract_features"]
