"""提取、标准化并保存 UCI HAR 的 39 维精简特征。

处理顺序严格遵循以下约束：

1. 读取官方训练分区和测试分区；
2. 先按受试者把官方训练分区拆成训练子集和验证子集；
3. 从九路、每路 128 点的窗口提取固定顺序的 39 维特征；
4. 只用训练子集拟合每一维的均值和标准差；
5. 用同一组参数标准化训练、验证和测试特征；
6. 保存三个数据分区、标准化参数和可复现实验的元数据。

这种顺序可避免验证集或测试集的分布信息进入训练阶段。输出特征顺序由
``feature_ref.feature_schema.FEATURE_NAMES`` 唯一确定，不能在不重新预处理和训练模型的
情况下改变。
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from feature_ref import FEATURE_DIM, FEATURE_NAMES, extract_features
from training.data import (
    ACTIVITY_NAMES,
    SIGNAL_CHANNELS,
    UCIHARDataset,
    iter_signal_dicts,
    load_uci_har_split,
    split_dataset_by_subject,
)

ProcessedSplitName = Literal["train", "val", "test"]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]

# 文件名集中定义，训练和评估模块可直接复用，避免在多处散落字符串常量。
SPLIT_FILENAMES: dict[ProcessedSplitName, str] = {
    "train": "train.npz",
    "val": "val.npz",
    "test": "test.npz",
}
STANDARDIZATION_FILENAME = "standardization.npz"
METADATA_FILENAME = "metadata.json"
PREPROCESSING_FORMAT_VERSION = 1


class ProcessedFeatureDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """保存标准化后的特征、标签和受试者编号。

    每个样本返回 ``(features, label, subject)``。其中 ``features`` 是形状为 ``(39,)``
    的 ``float32`` 张量，标签和受试者编号是 ``int64`` 标量。构造时会复制为连续数组，
    从磁盘加载后可直接交给 PyTorch DataLoader。
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        feature_array = np.asarray(features, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.int64)
        subject_array = np.asarray(subjects, dtype=np.int64)

        if feature_array.ndim != 2 or feature_array.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"features 的形状必须为 (样本数, {FEATURE_DIM})，"
                f"实际为 {feature_array.shape}"
            )
        sample_count = feature_array.shape[0]
        if sample_count == 0:
            raise ValueError("处理后的数据集不能为空")
        if label_array.shape != (sample_count,):
            raise ValueError(
                f"labels 的形状必须为 ({sample_count},)，实际为 {label_array.shape}"
            )
        if subject_array.shape != (sample_count,):
            raise ValueError(
                f"subjects 的形状必须为 ({sample_count},)，实际为 {subject_array.shape}"
            )
        if not np.all(np.isfinite(feature_array)):
            raise ValueError("features 包含 NaN 或 Inf")
        if np.any((label_array < 0) | (label_array >= len(ACTIVITY_NAMES))):
            raise ValueError("labels 必须位于 0～5")
        if np.any(subject_array <= 0):
            raise ValueError("subjects 必须是正整数")

        self.features = torch.from_numpy(np.ascontiguousarray(feature_array))
        self.labels = torch.from_numpy(np.ascontiguousarray(label_array))
        self.subjects = torch.from_numpy(np.ascontiguousarray(subject_array))

    def __len__(self) -> int:
        """返回处理后的窗口样本数。"""
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """返回一个 ``(39 维特征, 标签, 受试者编号)`` 三元组。"""
        return self.features[index], self.labels[index], self.subjects[index]


@dataclass(frozen=True)
class StandardizationParameters:
    """训练子集拟合得到的逐特征标准化参数。

    ``mean`` 和 ``scale`` 均使用 float64 保存，使检查点及 C++/量化转换阶段拥有
    足够精度。实际转换结果会统一输出 float32，与 MLP 输入类型一致。
    """

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    minimum_scale: float = 1e-8

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if mean.shape != (FEATURE_DIM,):
            raise ValueError(f"mean 的形状必须为 ({FEATURE_DIM},)，实际为 {mean.shape}")
        if scale.shape != (FEATURE_DIM,):
            raise ValueError(f"scale 的形状必须为 ({FEATURE_DIM},)，实际为 {scale.shape}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("标准化参数包含 NaN 或 Inf")
        if np.any(scale <= 0.0):
            raise ValueError("scale 的所有值都必须大于 0")
        if not np.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale 必须是有限正数")

        # frozen dataclass 仍允许在初始化校验阶段用 object.__setattr__ 固定数组副本。
        # 将数组设为只读，防止训练代码无意中改写已拟合的参数。
        mean = np.ascontiguousarray(mean.copy())
        scale = np.ascontiguousarray(scale.copy())
        mean.flags.writeable = False
        scale.flags.writeable = False
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(
        cls,
        training_features: np.ndarray,
        *,
        minimum_scale: float = 1e-8,
    ) -> StandardizationParameters:
        """只根据训练特征拟合均值和总体标准差。

        标准差采用 ``ddof=0``，与常用 ``StandardScaler`` 的总体方差约定一致。标准差
        小于 ``minimum_scale`` 的维度视为常量维度，其除数替换为 1，从而保留中心化
        结果并避免除零或数值被异常放大。
        """
        features = _validated_feature_matrix(training_features, name="training_features")
        if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
            raise ValueError("minimum_scale 必须是有限正数")

        # 输入矩阵形状为 (训练样本数, 39)。axis=0 表示沿“样本”方向统计，因此会为
        # 每个特征列分别得到一个均值和一个标准差，而不是用一组全局参数处理全部特征。
        # 在 float64 中累计均值和方差，可降低数千个窗口求和时的舍入误差。
        values = features.astype(np.float64, copy=False)
        # mean.shape == (39,)，mean[j] 是第 j 个特征在训练集上的均值。
        mean = np.mean(values, axis=0, dtype=np.float64)
        # raw_scale.shape == (39,)，采用总体标准差 ddof=0，与训练数据契约保持一致。
        raw_scale = np.std(values, axis=0, ddof=0, dtype=np.float64)
        # 近似常量列的标准差可能非常小；将其除数设为 1，避免除零或放大浮点噪声。
        scale = np.where(raw_scale < minimum_scale, 1.0, raw_scale)
        return cls(mean=mean, scale=scale, minimum_scale=minimum_scale)

    def transform(self, features: np.ndarray) -> FloatArray:
        """使用训练集参数执行 ``(x - mean) / scale``，返回 float32 数组。"""
        values = _validated_feature_matrix(features, name="features")
        # NumPy 会把形状为 (39,) 的 mean 和 scale 广播到矩阵的每一行：
        # standardized[i, j] = (values[i, j] - mean[j]) / scale[j]。
        # 因此 39 个特征维度全部进行独立的 Z-score 标准化；这不是将整行缩放到
        # [0, 1] 的 Min-Max 归一化，也不是把每条样本归一化成单位向量。
        standardized = (values.astype(np.float64, copy=False) - self.mean) / self.scale
        if not np.all(np.isfinite(standardized)):
            raise ValueError("标准化结果包含 NaN 或 Inf")
        return np.ascontiguousarray(standardized, dtype=np.float32)

    def inverse_transform(self, standardized_features: np.ndarray) -> FloatArray:
        """将标准化特征恢复到原始特征尺度，主要用于调试和结果核对。"""
        values = _validated_feature_matrix(
            standardized_features, name="standardized_features"
        )
        restored = values.astype(np.float64, copy=False) * self.scale + self.mean
        return np.ascontiguousarray(restored, dtype=np.float32)


@dataclass(frozen=True)
class PreprocessingResult:
    """一次完整预处理运行的内存结果与输出位置。"""

    datasets: dict[ProcessedSplitName, ProcessedFeatureDataset]
    standardization: StandardizationParameters
    output_dir: Path


def _validated_feature_matrix(features: np.ndarray, *, name: str) -> np.ndarray:
    """验证二维 39 维特征矩阵，保留输入精度供调用方选择计算类型。"""
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{name} 的形状必须为 (样本数, {FEATURE_DIM})，实际为 {values.shape}"
        )
    if values.shape[0] == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} 必须是数值数组")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    return values


def extract_dataset_features(
    dataset: UCIHARDataset,
    *,
    batch_size: int = 512,
) -> FloatArray:
    """从一个原始信号 Dataset 批量提取固定顺序的 39 维特征。

    输入 Dataset 中的 ``signals`` 形状为 ``(N, 9, 128)``：``N`` 是窗口数量，9 是
    ``SIGNAL_CHANNELS`` 定义的固定通道数，128 是每个通道在一个窗口中的采样点数。本函数
    将每个 ``(9, 128)`` 窗口转换为一个 ``(39,)`` 特征向量，最终返回形状为
    ``(N, 39)`` 的矩阵。

    单窗口计算由 :func:`feature_ref.extract_features` 完成，因此这里不重复实现均值、
    标准差、重力均值、模长统计和相关系数等公式。输出列顺序严格遵循
    ``FEATURE_NAMES``，而输出行顺序与原始 Dataset 完全相同，所以第 ``i`` 行特征仍然
    对应 ``dataset.labels[i]`` 和 ``dataset.subjects[i]``。

    ``batch_size`` 仅控制一次从 PyTorch 张量取出多少个窗口，以限制临时视图的大小；
    它不会执行训练意义上的随机打乱，也不会改变特征结果或样本顺序。这里不使用
    DataLoader 多进程，因为单窗口提取以 NumPy 小数组运算为主，进程间序列化九路原始
    信号通常比计算本身更昂贵。
    """
    if not isinstance(dataset, UCIHARDataset):
        raise TypeError("dataset 必须是 UCIHARDataset")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    # 一次性分配最终的 (N, 39) 连续矩阵，避免循环中反复 append 后再拼接产生额外副本。
    # np.empty 不会初始化内容，但下面会按照全局样本下标逐行完整写入。
    feature_matrix = np.empty((len(dataset), FEATURE_DIM), dtype=np.float32)

    # 外层循环只负责把大数据集切成若干连续小批次。例如 N=5800、batch_size=512 时，
    # 前 11 批各处理 512 个窗口，最后一批处理剩余窗口。连续切片保证样本原始顺序不变。
    for batch_start in range(0, len(dataset), batch_size):
        batch_stop = min(batch_start + batch_size, len(dataset))

        # signal_batch 的形状为 (当前批次窗口数, 9, 128)，仍是 CPU Tensor。
        # 这里只建立切片视图，不复制整个原始数据集。
        signal_batch = dataset.signals[batch_start:batch_stop]

        # iter_signal_dicts 按 SIGNAL_CHANNELS 的固定顺序，将每个 (9, 128) 窗口转换为
        # “通道名 -> 128 点 NumPy 数组”的映射。使用名称映射可以避免特征提取器依赖
        # 难以检查的数字通道下标，也能在通道缺失或顺序变化时尽早暴露问题。
        for local_index, signal_dict in enumerate(iter_signal_dicts(signal_batch)):
            # extract_features 对一个窗口计算固定顺序的 39 个 float32 特征。全局行号等于
            # 批次起始下标加批内下标，因此不会破坏特征与标签、受试者编号的一一对应关系。
            feature_matrix[batch_start + local_index] = extract_features(signal_dict)

    # extract_features 本身会检查单窗口输入和输出；这里再对完整矩阵做最终防御性检查，
    # 防止任何 NaN/Inf 进入标准化参数拟合，继而污染整个训练、验证和测试数据集。
    if not np.all(np.isfinite(feature_matrix)):
        raise RuntimeError("特征提取结果包含 NaN 或 Inf")
    return feature_matrix


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """先写同目录临时文件，再原子替换目标，避免中断后留下半个 NPZ。"""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary_path, path)
    finally:
        # 正常 replace 后临时文件已不存在；异常时尽量清理本次创建的残留文件。
        temporary_path.unlink(missing_ok=True)


def _atomic_save_json(path: Path, payload: dict[str, object]) -> None:
    """以 UTF-8 写入可读元数据，并通过原子替换保证文件完整性。"""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_processed_split(dataset: ProcessedFeatureDataset, path: str | Path) -> Path:
    """将一个处理后分区保存为压缩 NPZ，并嵌入特征 schema。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npz(
        output_path,
        features=dataset.features.numpy(),
        labels=dataset.labels.numpy(),
        subjects=dataset.subjects.numpy(),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    return output_path


def load_processed_split(path: str | Path) -> ProcessedFeatureDataset:
    """加载处理后的 NPZ，并确认其特征名称和当前代码完全一致。"""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到处理后的数据文件：{input_path}")
    try:
        with np.load(input_path, allow_pickle=False) as archive:
            required_keys = {"features", "labels", "subjects", "feature_names"}
            missing = sorted(required_keys.difference(archive.files))
            if missing:
                raise ValueError(f"文件缺少字段：{', '.join(missing)}")
            stored_names = tuple(str(name) for name in archive["feature_names"].tolist())
            if stored_names != FEATURE_NAMES:
                raise ValueError("文件中的特征名称或顺序与当前 FEATURE_NAMES 不一致")
            return ProcessedFeatureDataset(
                archive["features"], archive["labels"], archive["subjects"]
            )
    except (OSError, ValueError) as error:
        raise ValueError(f"无法加载处理后的数据文件 {input_path}：{error}") from error


def save_standardization_parameters(
    parameters: StandardizationParameters,
    path: str | Path,
) -> Path:
    """保存均值、缩放量、阈值及对应的固定特征顺序。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npz(
        output_path,
        mean=parameters.mean,
        scale=parameters.scale,
        minimum_scale=np.asarray(parameters.minimum_scale, dtype=np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    return output_path


def load_standardization_parameters(path: str | Path) -> StandardizationParameters:
    """加载标准化参数，并拒绝旧 schema 或字段不完整的文件。"""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到标准化参数文件：{input_path}")
    try:
        with np.load(input_path, allow_pickle=False) as archive:
            required_keys = {"mean", "scale", "minimum_scale", "feature_names"}
            missing = sorted(required_keys.difference(archive.files))
            if missing:
                raise ValueError(f"文件缺少字段：{', '.join(missing)}")
            stored_names = tuple(str(name) for name in archive["feature_names"].tolist())
            if stored_names != FEATURE_NAMES:
                raise ValueError("标准化参数的特征名称或顺序与当前 FEATURE_NAMES 不一致")
            minimum_scale = float(np.asarray(archive["minimum_scale"]).item())
            return StandardizationParameters(
                mean=archive["mean"],
                scale=archive["scale"],
                minimum_scale=minimum_scale,
            )
    except (OSError, ValueError) as error:
        raise ValueError(f"无法加载标准化参数文件 {input_path}：{error}") from error


def preprocess_uci_har(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    validation_subjects: Sequence[int] | None = None,
    extraction_batch_size: int = 64,
    minimum_scale: float = 1e-8,
    overwrite: bool = False,
) -> PreprocessingResult:
    """执行完整预处理并将结果保存到 ``output_dir``。

    默认拒绝覆盖任何已有目标文件，避免不同随机种子或划分配置的实验结果被静默替换。
    确认需要更新产物时显式传入 ``overwrite=True``。
    """
    destination = Path(output_dir).expanduser()
    output_paths = [destination / filename for filename in SPLIT_FILENAMES.values()]
    output_paths.extend(
        (destination / STANDARDIZATION_FILENAME, destination / METADATA_FILENAME)
    )
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        joined_paths = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"输出文件已存在；如需覆盖请设置 overwrite=True：{joined_paths}")

    official_train = load_uci_har_split(dataset_root, "train")
    raw_test = load_uci_har_split(dataset_root, "test")
    raw_train, raw_validation = split_dataset_by_subject(
        official_train,
        validation_fraction=validation_fraction,
        seed=seed,
        validation_subjects=validation_subjects,
    )

    # 必须先按受试者完成训练/验证划分，再拟合标准化参数。同一受试者的所有重叠窗口只会
    # 出现在一个子集中，避免个人运动模式和相邻窗口泄漏到验证结果中。
    raw_datasets: dict[ProcessedSplitName, UCIHARDataset] = {
        "train": raw_train,
        "val": raw_validation,
        "test": raw_test,
    }

    # 分别把三个原始信号 Dataset 从 (N, 9, 128) 转换为 (N, 39) 特征矩阵。
    # 测试集虽然也需要提取特征，但它只参与最终变换和评估，绝不参与均值/方差拟合。
    extracted = {
        split: extract_dataset_features(dataset, batch_size=extraction_batch_size)
        for split, dataset in raw_datasets.items()
    }

    # 标准化参数只根据训练子集计算。验证集和测试集随后复用同一组 mean/scale，模拟部署时
    # 只能使用训练阶段已保存参数处理未知样本的真实条件。
    standardization = StandardizationParameters.fit(
        extracted["train"], minimum_scale=minimum_scale
    )

    processed_datasets: dict[ProcessedSplitName, ProcessedFeatureDataset] = {}
    for split, raw_dataset in raw_datasets.items():
        # train、val、test 都调用同一个 transform，因此全部使用上面仅由训练集拟合的
        # 39 组 mean/scale；验证集和测试集不会重新计算自己的标准化参数。
        # transform 保持矩阵行序不变，因此标准化后的第 i 行仍与对应原始 Dataset 的
        # labels[i] 和 subjects[i] 对齐。最终 Dataset 每项返回 (39维特征, 标签, 受试者)。
        processed_datasets[split] = ProcessedFeatureDataset(
            standardization.transform(extracted[split]),
            raw_dataset.labels.numpy(),
            raw_dataset.subjects.numpy(),
        )

    destination.mkdir(parents=True, exist_ok=True)
    for split, dataset in processed_datasets.items():
        save_processed_split(dataset, destination / SPLIT_FILENAMES[split])
    save_standardization_parameters(
        standardization, destination / STANDARDIZATION_FILENAME
    )

    # 保存“实际”验证受试者，而不只保存输入参数。随机划分时这组编号是复现实验的关键。
    actual_validation_subjects = sorted(
        set(processed_datasets["val"].subjects.tolist())
    )
    metadata: dict[str, object] = {
        "format_version": PREPROCESSING_FORMAT_VERSION,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "signal_channels": list(SIGNAL_CHANNELS),
        "activity_names": list(ACTIVITY_NAMES),
        "label_encoding": "zero_based",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "validation_subjects": actual_validation_subjects,
        "standardization": {
            "fit_split": "train",
            "variance_ddof": 0,
            "minimum_scale": minimum_scale,
            "constant_feature_scale": 1.0,
        },
        "splits": {
            split: {
                "file": SPLIT_FILENAMES[split],
                "sample_count": len(dataset),
                "subjects": sorted(set(dataset.subjects.tolist())),
            }
            for split, dataset in processed_datasets.items()
        },
    }
    _atomic_save_json(destination / METADATA_FILENAME, metadata)

    return PreprocessingResult(
        datasets=processed_datasets,
        standardization=standardization,
        output_dir=destination,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/UCI HAR Dataset"),
        help="解压后的 UCI HAR Dataset 根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="处理结果输出目录",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-subjects",
        type=int,
        nargs="+",
        default=None,
        help="固定验证受试者编号；省略时根据比例和随机种子选择",
    )
    parser.add_argument("--extraction-batch-size", type=int, default=512)
    parser.add_argument("--minimum-scale", type=float, default=1e-8)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有处理结果")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；成功后打印各分区样本数和输出目录。"""
    arguments = _build_argument_parser().parse_args(argv)
    result = preprocess_uci_har(
        arguments.dataset_root,
        arguments.output_dir,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        validation_subjects=arguments.validation_subjects,
        extraction_batch_size=arguments.extraction_batch_size,
        minimum_scale=arguments.minimum_scale,
        overwrite=arguments.overwrite,
    )
    counts = ", ".join(
        f"{split}={len(dataset)}" for split, dataset in result.datasets.items()
    )
    print(f"预处理完成：{counts}")
    print(f"输出目录：{result.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METADATA_FILENAME",
    "PREPROCESSING_FORMAT_VERSION",
    "SPLIT_FILENAMES",
    "STANDARDIZATION_FILENAME",
    "PreprocessingResult",
    "ProcessedFeatureDataset",
    "StandardizationParameters",
    "extract_dataset_features",
    "load_processed_split",
    "load_standardization_parameters",
    "main",
    "preprocess_uci_har",
    "save_processed_split",
    "save_standardization_parameters",
]
