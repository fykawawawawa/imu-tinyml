"""精简 IMU 时域特征向量的固定结构定义。

本模块定义 Python 特征提取器、特征标准化参数和 MLP 输入层之间的共同契约。特征名称
或顺序一旦写入训练模型，修改后必须重新生成数据并训练模型。C++ 端若实现同一套特征，
也必须遵循此处定义的顺序。

索引约定：代码中的 ``FEATURE_INDEX`` 从 0 开始；文档中常用的“第 1～39 维”从 1
开始。两者指向同一组特征，只相差 1。
"""

from __future__ import annotations

from types import MappingProxyType

# UCI HAR 以 50 Hz 采样，并使用 2.56 秒窗口，因此每个通道为 50 × 2.56 = 128 点。
WINDOW_SIZE = 128

# 参与动态统计的 6 个身体运动通道。
# 此处顺序决定前 24 维的分组顺序：先加速度 XYZ，再角速度 XYZ。
DYNAMIC_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)

# 提取器还需要总加速度 XYZ，以便通过 total_acc - body_acc 恢复重力分量。
# 因此一个窗口必须提供 9 个通道；额外通道不会参与本版 39 维特征计算。
REQUIRED_CHANNELS = DYNAMIC_CHANNELS + (
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)

# 每个动态轴依次产生 4 维。RMS 未保留，因为在 ddof=0 时可由 mean 和 std 推导；RMS^2 = mean^2 + std^2
# min/max 也未同时保留，而是用 peak_to_peak 表示完整振幅范围。
AXIS_STAT_NAMES = ("mean", "std", "abs_mean", "peak_to_peak")

# 每组三轴模长依次产生 3 维，用于表达较少依赖手机坐标方向的整体运动强度。
MAGNITUDE_STAT_NAMES = ("mean", "std", "max")


def _build_feature_names() -> tuple[str, ...]:
    """按提取器的拼接顺序生成全部特征名称。

    返回 tuple 而不是 list，避免调用方意外修改全局 schema。分组如下：

    - 索引 0～23：6 个动态轴 × 4 个统计量；
    - 索引 24～26：重力 XYZ 均值；
    - 索引 27～32：加速度/角速度模长 × 3 个统计量；
    - 索引 33～38：加速度/角速度的 XY、XZ、YZ 相关系数。
    """
    names: list[str] = []

    # 动态轴统计：通道为外层、统计量为内层，和 extractor.py 的循环完全一致。
    # 6 个动态轴 × 4 个统计量；
    for channel in DYNAMIC_CHANNELS:
        names.extend(f"{channel}_{stat}" for stat in AXIS_STAT_NAMES)

    # 重力只保留窗口均值，用于区分站、坐、躺等静态姿态。
    names.extend(
        (
            "gravity_mean_x",
            "gravity_mean_y",
            "gravity_mean_z",
        )
    )

    # 模长顺序固定为身体加速度在前、角速度在后。
    for group in ("body_acc_mag", "body_gyro_mag"):
        names.extend(f"{group}_{stat}" for stat in MAGNITUDE_STAT_NAMES)

    # 每组三轴信号只存在 3 个不重复的轴对：XY、XZ 和 YZ。 共6个相关性
    names.extend(
        (
            "body_acc_corr_xy",
            "body_acc_corr_xz",
            "body_acc_corr_yz",
            "body_gyro_corr_xy",
            "body_gyro_corr_xz",
            "body_gyro_corr_yz",
        )
    )
    return tuple(names)


# FEATURE_NAMES 用于保存列名；FEATURE_DIM 用于模型输入维度检查。
FEATURE_NAMES = _build_feature_names()
FEATURE_DIM = len(FEATURE_NAMES)

# MappingProxyType 提供只读视图，防止运行时误改名称到索引的映射。
# FEATURE_INDEX 使用从 0 开始的索引，可直接索引 NumPy 特征向量。
FEATURE_INDEX = MappingProxyType({name: index for index, name in enumerate(FEATURE_NAMES)})

# 本项目当前设计明确要求 39 维。若修改列表后没有同步设计，在模块导入时立即失败，
# 比等到训练或部署时才发现 MLP 输入维度不匹配更容易定位。
assert FEATURE_DIM == 39


# 只导出需要跨模块共享的 schema 常量；内部构建函数不属于公共 API。
__all__ = [
    "AXIS_STAT_NAMES",
    "DYNAMIC_CHANNELS",
    "FEATURE_DIM",
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "MAGNITUDE_STAT_NAMES",
    "REQUIRED_CHANNELS",
    "WINDOW_SIZE",
]
