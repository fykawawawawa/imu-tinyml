"""单个 IMU 窗口的基础时域特征计算。

本模块只实现与具体数据集文件布局无关的数学运算，不负责决定窗口长度或特征拼接
顺序。所有入口先将数据转换为 float64 并检查有限性；最终模型需要的 float32 转换统一
由 ``extractor.py`` 完成。本模块也可作为 C++ 特征实现的数值参考。

数值约定：标准差使用总体标准差（``ddof=0``）；相关系数遇到零方差信号时返回 0；
所有函数都拒绝 NaN、正无穷和负无穷。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# 公共类型别名：本模块内部统一使用一维 float64 NumPy 数组。
FloatArray = NDArray[np.float64]

# 相关系数分母阈值。若两个中心化向量的范数乘积不大于该值，则认为至少有一个信号
# 缺少有效波动，相关系数按项目约定返回 0。
CORRELATION_EPSILON = 1.0e-8


def as_float64_vector(values: ArrayLike, *, name: str = "signal") -> FloatArray:
    """把输入转换为非空、有限的一维 float64 信号。

    参数：
        values: 可被 NumPy 转换的数组状对象，例如 list、tuple 或 ndarray。
        name: 仅用于异常信息，通常传入实际通道名以便定位坏数据。

    返回：
        一维 float64 NumPy 数组。本函数及其他统计函数都不会修改其内容。

    异常：
        ValueError: 输入不是一维、长度为零，或包含 NaN/Inf。

    注意：
        本函数只验证通用的“一维且有限”条件。UCI HAR 固定 128 点的约束由
        ``extractor._validated_window`` 检查，以便这些数学函数也能单独测试。
    """
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} contains NaN or infinity")
    return signal


def axis_statistics(values: ArrayLike) -> tuple[float, float, float, float]:
    """计算一个动态轴的 4 个统计量。

    返回值顺序严格为 ``(mean, std, abs_mean, peak_to_peak)``：

    - ``mean`` 描述窗口内的平均偏置；
    - ``std`` 描述围绕均值的波动强度，使用总体标准差；
    - ``abs_mean`` 描述不区分正负方向的平均活动强度；
    - ``peak_to_peak`` 等于最大值减最小值，描述完整振幅范围。

    返回 Python float，而不是 NumPy 标量，便于特征列表拼接和序列化。
    """
    signal = as_float64_vector(values)
    return (
        float(np.mean(signal)),
        # ddof=0 表示总体标准差；C++ 实现必须使用相同定义。
        float(np.std(signal, ddof=0)),
        float(np.mean(np.abs(signal))),
        float(np.ptp(signal)),
    )


def vector_magnitude(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> FloatArray:
    """逐采样点计算三轴信号的欧氏模长。

    第 n 个输出为 ``sqrt(x[n]**2 + y[n]**2 + z[n]**2)``。模长始终非负，且在
    手机坐标系发生旋转时比单轴数值更稳定。输出单位与输入一致：加速度输入得到 g，
    角速度输入得到 rad/s。

    三个输入必须都是长度相同的一维有限信号，否则抛出 ValueError。
    """
    x_signal = as_float64_vector(x, name="x")
    y_signal = as_float64_vector(y, name="y")
    z_signal = as_float64_vector(z, name="z")
    if not (x_signal.shape == y_signal.shape == z_signal.shape):
        raise ValueError("x, y, and z must have the same shape")
    return np.sqrt(x_signal * x_signal + y_signal * y_signal + z_signal * z_signal)


def magnitude_statistics(values: ArrayLike) -> tuple[float, float, float]:
    """计算模长信号的均值、总体标准差和最大值。

    返回值顺序严格为 ``(mean, std, max)``。均值描述整体活动强度，标准差描述窗口内
    强度变化，最大值保留短时峰值信息。这里不再计算 abs_mean，因为模长本身非负，
    所以它与 mean 完全相同。
    """
    signal = as_float64_vector(values, name="magnitude")
    return (
        float(np.mean(signal)),
        float(np.std(signal, ddof=0)),
        float(np.max(signal)),
    )


def safe_correlation(
    x: ArrayLike,
    y: ArrayLike,
    *,
    epsilon: float = CORRELATION_EPSILON,
) -> float:
    """计算两个等长信号的皮尔逊相关系数。

    参数：
        x: 第一条一维有限信号。
        y: 第二条一维有限信号，长度必须与 x 相同。
        epsilon: 判断分母接近零的阈值，默认使用项目统一常量。

    返回：
        ``[-1, 1]`` 内的 Python float。正值表示同向变化，负值表示反向变化；任一
        输入近似恒定时返回 0，因为此时相关方向没有可靠定义。

    计算使用中心化向量点积除以范数乘积，与常见的皮尔逊相关系数定义等价。由于分子
    和分母中的样本数量因子会抵消，因此不涉及 ddof 选择。
    """
    x_signal = as_float64_vector(x, name="x")
    y_signal = as_float64_vector(y, name="y")
    if x_signal.shape != y_signal.shape:
        raise ValueError("x and y must have the same shape")

    # 皮尔逊相关系数的向量形式为：
    #
    #                   Σ[(xᵢ - x̄)(yᵢ - ȳ)]
    #   corr(x, y) = ---------------------------------
    #                √Σ(xᵢ - x̄)² · √Σ(yᵢ - ȳ)²
    #
    # 第一步先从每个采样点减去所在窗口的均值，得到中心化向量。这样会消除各轴的
    # 直流偏置，相关系数只描述两个轴围绕各自均值是否同步变化。例如两个波形即使
    # 分别叠加不同的常量偏置，中心化后的相关系数仍然相同。
    x_centered = x_signal - np.mean(x_signal)
    y_centered = y_signal - np.mean(y_signal)

    # np.linalg.norm(v) = √Σvᵢ²，因此两个范数的乘积正好对应公式分母，默认求二范数。
    # 分母也可以理解为两个中心化信号“波动大小”的乘积，用来消除绝对振幅尺度：
    # 若 y 是 x 的正比例缩放，相关系数仍为 1；若是负比例缩放，则为 -1。
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))

    # 恒定信号中心化后全为 0，其范数和分母也为 0，数学上的相关系数未定义。
    # 近似恒定信号的分母虽然可能非零，但直接相除会放大浮点误差。因此分母不大于
    # epsilon 时按项目约定返回 0，表示“没有可可靠估计的轴间相关性”，同时避免 NaN。
    if denominator <= epsilon:
        return 0.0

    # np.dot(x_centered, y_centered) = Σ[(xᵢ - x̄)(yᵢ - ȳ)]，即公式分子。
    # 分子为正表示两个轴倾向同向变化，为负表示倾向反向变化，接近 0 表示没有明显
    # 的线性共同变化。这里不需要显式除以样本数 N，因为分子和分母中的 N 会相互抵消。
    correlation = float(np.dot(x_centered, y_centered) / denominator)
    # 浮点舍入可能产生略微超出 [-1, 1] 的结果，因此显式截断。
    return float(np.clip(correlation, -1.0, 1.0))


# 明确公共 API，避免调用方依赖仅供内部使用的类型别名。
__all__ = [
    "CORRELATION_EPSILON",
    "as_float64_vector",
    "axis_statistics",
    "magnitude_statistics",
    "safe_correlation",
    "vector_magnitude",
]
