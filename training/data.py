"""UCI HAR 九路惯性信号的数据读取与批处理。

本模块只负责把官方文本文件可靠地转换成 PyTorch 数据接口，不在这里提取 39 维特征
或执行标准化。每个样本返回 ``(signals, label, subject)``：

* ``signals`` 是形状为 ``(9, 128)`` 的 ``float32`` 张量；
* ``label`` 是 ``int64`` 标量，并已从官方的 ``1～6`` 转换为交叉熵所需的 ``0～5``；
* ``subject`` 是 ``int64`` 标量，保留官方受试者编号。

九路信号的顺序与 :mod:`feature_ref.feature_schema` 中的 ``REQUIRED_CHANNELS`` 完全一致，
后续特征提取代码可通过通道名称和该顺序进行一一映射。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from feature_ref.feature_schema import REQUIRED_CHANNELS, WINDOW_SIZE

# 使用 tuple 固定通道顺序，防止训练过程中被意外修改。
SIGNAL_CHANNELS = REQUIRED_CHANNELS

# 下标就是转换后的训练标签，便于评估阶段直接将预测结果还原为活动名称。
ACTIVITY_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)

SplitName = Literal["train", "test"]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


class UCIHARDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """在内存中保存 UCI HAR 窗口、标签和受试者编号。

    参数：
        signals: 形状为 ``(样本数, 9, 128)`` 的浮点数组。第二维的排列必须遵循
            ``SIGNAL_CHANNELS``。
        labels: 形状为 ``(样本数,)`` 的整数数组，取值必须已经是 ``0～5``。
        subjects: 形状为 ``(样本数,)`` 的正整数数组，表示每个窗口对应的受试者。

    数据在构造时统一复制为连续内存。这样，即使调用者传入的是切片视图，生成的张量也
    不会依赖外部数组的生命周期；DataLoader 多进程读取时也具有稳定、明确的内存布局。
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
    ) -> None:
        signals_array = np.asarray(signals, dtype=np.float32)
        labels_array = np.asarray(labels, dtype=np.int64)
        subjects_array = np.asarray(subjects, dtype=np.int64)

        expected_tail = (len(SIGNAL_CHANNELS), WINDOW_SIZE)
        if signals_array.ndim != 3 or signals_array.shape[1:] != expected_tail:
            raise ValueError(
                "signals 的形状必须为 "
                f"(样本数, {expected_tail[0]}, {expected_tail[1]})，"
                f"实际为 {signals_array.shape}"
            )
        sample_count = signals_array.shape[0]
        if labels_array.shape != (sample_count,):
            raise ValueError(
                f"labels 的形状必须为 ({sample_count},)，实际为 {labels_array.shape}"
            )
        if subjects_array.shape != (sample_count,):
            raise ValueError(
                f"subjects 的形状必须为 ({sample_count},)，实际为 {subjects_array.shape}"
            )
        if sample_count == 0:
            raise ValueError("数据集不能为空")
        if not np.all(np.isfinite(signals_array)):
            raise ValueError("signals 包含 NaN 或 Inf")
        if np.any((labels_array < 0) | (labels_array >= len(ACTIVITY_NAMES))):
            raise ValueError("labels 必须位于 0～5")
        if np.any(subjects_array <= 0):
            raise ValueError("subjects 必须是正整数")

        # torch.from_numpy 不复制内存；先在这里生成独立的 C 连续数组，后续索引更高效。
        self.signals = torch.from_numpy(np.ascontiguousarray(signals_array))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels_array))
        self.subjects = torch.from_numpy(np.ascontiguousarray(subjects_array))

    def __len__(self) -> int:
        """返回窗口样本数。"""
        return self.signals.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """返回一个 ``(九路信号, 标签, 受试者编号)`` 三元组。"""
        return self.signals[index], self.labels[index], self.subjects[index]

    def subset(self, indices: Sequence[int] | NDArray[np.integer]) -> UCIHARDataset:
        """根据样本下标创建独立子集，同时保留完整的 Dataset 接口。

        此处不使用 ``torch.utils.data.Subset``，因为后续预处理需要直接访问子集的
        ``signals``、``labels`` 和 ``subjects`` 属性来拟合训练集统计量。
        """
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.ndim != 1:
            raise ValueError(f"indices 必须是一维数组，实际形状为 {index_array.shape}")
        if index_array.size == 0:
            raise ValueError("子集至少需要包含一个样本")
        if np.any((index_array < 0) | (index_array >= len(self))):
            raise IndexError("indices 包含超出数据集范围的下标")

        # 高级索引会产生副本，正好满足子集不依赖原始大数组的设计目标。
        return UCIHARDataset(
            self.signals[index_array].numpy(),
            self.labels[index_array].numpy(),
            self.subjects[index_array].numpy(),
        )


def _require_file(path: Path) -> Path:
    """确认输入文件存在，并在失败时给出具体路径。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 UCI HAR 数据文件：{path}")
    return path


def _load_signal_file(path: Path) -> FloatArray:
    """读取单个信号文件并校验每行恰好包含 128 个采样点。"""
    try:
        # ndmin=2 可保证只有一行的测试文件仍返回二维数组。
        values = np.loadtxt(_require_file(path), dtype=np.float32, ndmin=2)
    except ValueError as error:
        raise ValueError(f"无法解析信号文件 {path}：{error}") from error

    if values.shape[1] != WINDOW_SIZE:
        raise ValueError(
            f"信号文件 {path} 每行应有 {WINDOW_SIZE} 个采样点，"
            f"实际形状为 {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"信号文件 {path} 包含 NaN 或 Inf")
    return values


def _load_integer_vector(path: Path, *, name: str) -> IntArray:
    """读取标签或受试者编号，并拒绝小数、NaN 等非整数内容。"""
    try:
        # 先按 float64 读取再检查整数性，避免 ``dtype=int64`` 静默截断 ``1.5``。
        raw_values = np.loadtxt(_require_file(path), dtype=np.float64, ndmin=1)
    except ValueError as error:
        raise ValueError(f"无法解析{name}文件 {path}：{error}") from error

    if raw_values.ndim != 1:
        raise ValueError(f"{name}文件 {path} 必须只有一列，实际形状为 {raw_values.shape}")
    if not np.all(np.isfinite(raw_values)) or not np.all(raw_values == np.floor(raw_values)):
        raise ValueError(f"{name}文件 {path} 必须只包含有限整数")
    return raw_values.astype(np.int64, copy=False)


def load_uci_har_split(dataset_root: str | Path, split: SplitName) -> UCIHARDataset:
    """批量读取 UCI HAR 的一个官方分区。

    参数：
        dataset_root: 解压后的 ``UCI HAR Dataset`` 根目录，而不是其 ``train`` 子目录。
        split: 只能是 ``"train"`` 或 ``"test"``。

    返回：
        完整载入内存的 :class:`UCIHARDataset`。信号形状为 ``(N, 9, 128)``，标签已
        转换为 ``0～5``，受试者编号保持官方值。

    异常：
        ValueError: 分区名、文件内容、形状、标签范围或各文件样本数不合法。
        FileNotFoundError: 任一必需文件不存在。
    """
    if split not in ("train", "test"):
        raise ValueError(f"split 只能是 'train' 或 'test'，实际为 {split!r}")

    root = Path(dataset_root).expanduser()
    split_dir = root / split
    signal_dir = split_dir / "Inertial Signals"

    # 九个文件按固定 schema 顺序读取；同一行号在所有文件中代表同一个时间窗口。
    channel_arrays: list[FloatArray] = []
    for channel in SIGNAL_CHANNELS:
        channel_path = signal_dir / f"{channel}_{split}.txt"
        channel_arrays.append(_load_signal_file(channel_path))

    labels_official = _load_integer_vector(split_dir / f"y_{split}.txt", name="标签")
    subjects = _load_integer_vector(split_dir / f"subject_{split}.txt", name="受试者编号")

    # 先列出所有计数再统一检查，报错信息能直接指出是哪一路信号发生错位。
    sample_counts = {
        **{
            channel: array.shape[0]
            for channel, array in zip(SIGNAL_CHANNELS, channel_arrays, strict=True)
        },
        "labels": labels_official.size,
        "subjects": subjects.size,
    }
    if len(set(sample_counts.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in sample_counts.items())
        raise ValueError(f"{split} 分区的九路信号、标签和受试者样本数不一致：{details}")

    if np.any((labels_official < 1) | (labels_official > len(ACTIVITY_NAMES))):
        invalid = np.unique(labels_official[(labels_official < 1) | (labels_official > 6)])
        raise ValueError(f"官方标签必须位于 1～6，发现非法值：{invalid.tolist()}")
    if np.any(subjects <= 0):
        invalid = np.unique(subjects[subjects <= 0])
        raise ValueError(f"受试者编号必须为正整数，发现非法值：{invalid.tolist()}")

    # 在 axis=1 拼接通道，得到模型约定的 (样本, 通道, 时间) 布局。
    signals = np.stack(channel_arrays, axis=1)
    labels_zero_based = labels_official - 1
    return UCIHARDataset(signals, labels_zero_based, subjects)


def split_dataset_by_subject(
    dataset: UCIHARDataset,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    validation_subjects: Sequence[int] | None = None,
) -> tuple[UCIHARDataset, UCIHARDataset]:
    """按受试者将官方训练集拆成训练子集和验证子集。

    相邻 UCI HAR 窗口有 50% 重叠，所以不能按窗口随机拆分。此函数保证一个受试者的
    所有窗口只出现在一个子集中，从根源上避免相邻窗口或个人运动模式泄漏。

    ``validation_subjects`` 可用于完全复现实验；未指定时，会用 ``seed`` 打乱受试者
    编号并选择接近 ``validation_fraction`` 的人数。人数采用四舍五入，且训练和验证
    两侧都至少保留一名受试者。
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction 必须严格位于 0 和 1 之间")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数")

    all_subjects = np.unique(dataset.subjects.numpy())
    if all_subjects.size < 2:
        raise ValueError("按受试者划分至少需要两名不同受试者")

    # 验证集数量
    if validation_subjects is None:
        validation_count = int(np.floor(all_subjects.size * validation_fraction + 0.5))
        validation_count = min(max(validation_count, 1), all_subjects.size - 1)
        shuffled_subjects = np.random.default_rng(seed).permutation(all_subjects)
        selected_subjects = np.sort(shuffled_subjects[:validation_count])
    else:
        raw_subjects = np.asarray(validation_subjects)
        if raw_subjects.ndim != 1 or raw_subjects.size == 0:
            raise ValueError("validation_subjects 必须是一维非空序列")
        try:
            numeric_subjects = raw_subjects.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("validation_subjects 必须只包含整数") from error
        if not np.all(np.isfinite(numeric_subjects)) or not np.all(
            numeric_subjects == np.floor(numeric_subjects)
        ):
            raise ValueError("validation_subjects 必须只包含有限整数")
        selected_subjects = numeric_subjects.astype(np.int64)
        if np.unique(selected_subjects).size != selected_subjects.size:
            raise ValueError("validation_subjects 不能包含重复编号")
        unknown = np.setdiff1d(selected_subjects, all_subjects)
        if unknown.size:
            raise ValueError(f"验证受试者不在数据集中：{unknown.tolist()}")
        if selected_subjects.size == all_subjects.size:
            raise ValueError("验证集不能包含全部受试者")

    validation_mask = np.isin(dataset.subjects.numpy(), selected_subjects)
    training_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    return dataset.subset(training_indices), dataset.subset(validation_indices)


def _seed_worker(worker_id: int) -> None:
    """为每个 DataLoader 工作进程设置 NumPy 随机种子。

    当前 Dataset 本身没有随机变换，但提前固定种子可以保证后续加入数据增强后，多进程
    加载仍然可复现。PyTorch 会在调用本函数前为每个工作进程设置不同的 initial_seed。
    """
    del worker_id  # 工作进程差异已经包含在 torch.initial_seed() 中。
    np.random.seed(torch.initial_seed() % (2**32))


def _make_data_loader(
    dataset: UCIHARDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """使用统一参数构造一个可复现的 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # 只有训练集会打乱，但三个加载器都传入 generator，使未来随机变换行为一致。
        generator=generator,
        worker_init_fn=_seed_worker,
        # 没有工作进程时 PyTorch 不允许 persistent_workers=True。
        persistent_workers=num_workers > 0,
    )


def create_data_loaders(
    dataset_root: str | Path,
    *,
    batch_size: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 42,
    validation_subjects: Sequence[int] | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> dict[str, DataLoader[tuple[Tensor, Tensor, Tensor]]]:
    """读取完整数据并创建训练、验证、测试 DataLoader。

    训练集由官方 ``train`` 分区按受试者继续拆分；测试集始终使用官方 ``test`` 分区，
    不参与验证集选择。只有训练 DataLoader 会打乱顺序，验证和测试保持官方行序。

    参数：
        dataset_root: 解压后的 ``UCI HAR Dataset`` 根目录。
        batch_size: 每个批次的最大样本数，必须为正整数。
        validation_fraction: 未显式指定验证受试者时，验证受试者占训练受试者的比例。
        seed: 控制验证受试者选择、训练批次打乱及工作进程随机状态。
        validation_subjects: 可选的固定验证受试者编号序列。
        num_workers: DataLoader 子进程数；Windows 或调试环境可保持默认值 0。
        pin_memory: 是否使用页锁定内存。为 ``None`` 时按 CUDA 是否可用自动选择。
        drop_last: 训练时是否丢弃最后一个不足 ``batch_size`` 的批次。验证和测试绝不丢弃。

    返回：
        含 ``"train"``、``"val"``、``"test"`` 三个键的字典。每个批次均为
        ``(signals, labels, subjects)``，形状依次为 ``(B, 9, 128)``、``(B,)``、
        ``(B,)``。
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers 必须是非负整数")
    if pin_memory is not None and not isinstance(pin_memory, bool):
        raise ValueError("pin_memory 必须是布尔值或 None")
    if not isinstance(drop_last, bool):
        raise ValueError("drop_last 必须是布尔值")

    official_train = load_uci_har_split(dataset_root, "train")
    test_dataset = load_uci_har_split(dataset_root, "test")
    train_dataset, validation_dataset = split_dataset_by_subject(
        official_train,
        validation_fraction=validation_fraction,
        seed=seed,
        validation_subjects=validation_subjects,
    )

    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    common_options = {
        "batch_size": batch_size,
        "seed": seed,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    return {
        "train": _make_data_loader(
            train_dataset, shuffle=True, drop_last=drop_last, **common_options
        ),
        "val": _make_data_loader(
            validation_dataset, shuffle=False, drop_last=False, **common_options
        ),
        "test": _make_data_loader(
            test_dataset, shuffle=False, drop_last=False, **common_options
        ),
    }


def iter_signal_dicts(signals: Tensor) -> Iterator[dict[str, NDArray[np.float32]]]:
    """把一批九路张量逐窗口转换为特征提取器需要的通道字典。

    该辅助函数让后续 ``training/preprocess.py`` 无需重复维护通道索引。输入可以是单个
    ``(9, 128)`` 窗口或一批 ``(B, 9, 128)`` 窗口；张量必须位于 CPU，因为 NumPy
    特征参考实现只在 CPU 上运行。
    """
    if signals.device.type != "cpu":
        raise ValueError("转换为 NumPy 通道字典前，signals 必须位于 CPU")
    if signals.ndim == 2:
        signals = signals.unsqueeze(0)
    expected_tail = (len(SIGNAL_CHANNELS), WINDOW_SIZE)
    if signals.ndim != 3 or tuple(signals.shape[1:]) != expected_tail:
        raise ValueError(
            f"signals 的形状必须为 (B, {expected_tail[0]}, {expected_tail[1]})，"
            f"实际为 {tuple(signals.shape)}"
        )

    values = signals.detach().numpy()
    for window in values:
        yield {
            channel: window[channel_index]
            for channel_index, channel in enumerate(SIGNAL_CHANNELS)
        }


__all__ = [
    "ACTIVITY_NAMES",
    "SIGNAL_CHANNELS",
    "UCIHARDataset",
    "create_data_loaders",
    "iter_signal_dicts",
    "load_uci_har_split",
    "split_dataset_by_subject",
]
