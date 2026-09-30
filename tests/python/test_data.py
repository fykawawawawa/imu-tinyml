"""UCI HAR 数据读取和受试者级拆分测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from training.data import (
    SIGNAL_CHANNELS,
    UCIHARDataset,
    create_data_loaders,
    iter_signal_dicts,
    load_uci_har_split,
    split_dataset_by_subject,
)


def _write_split(root: Path, split: str, labels: list[int], subjects: list[int]) -> None:
    """写入体积很小、但目录结构完整的伪 UCI HAR 分区。"""
    signal_dir = root / split / "Inertial Signals"
    signal_dir.mkdir(parents=True)
    sample_count = len(labels)
    for channel_index, channel in enumerate(SIGNAL_CHANNELS):
        values = np.full((sample_count, 128), channel_index + 0.25, dtype=np.float32)
        np.savetxt(signal_dir / f"{channel}_{split}.txt", values)
    np.savetxt(root / split / f"y_{split}.txt", labels, fmt="%d")
    np.savetxt(root / split / f"subject_{split}.txt", subjects, fmt="%d")


@pytest.fixture
def tiny_uci_har(tmp_path: Path) -> Path:
    root = tmp_path / "UCI HAR Dataset"
    _write_split(root, "train", [1, 2, 3, 4, 5, 6], [1, 1, 2, 2, 3, 3])
    _write_split(root, "test", [6, 1], [4, 4])
    return root


def test_load_split_returns_expected_tensors(tiny_uci_har: Path) -> None:
    dataset = load_uci_har_split(tiny_uci_har, "train")

    assert len(dataset) == 6
    assert dataset.signals.shape == (6, 9, 128)
    assert dataset.signals.dtype == torch.float32
    assert dataset.labels.tolist() == [0, 1, 2, 3, 4, 5]
    assert dataset.subjects.tolist() == [1, 1, 2, 2, 3, 3]
    assert dataset.signals[0, 8, 0].item() == pytest.approx(8.25)


def test_subject_split_has_no_subject_leakage(tiny_uci_har: Path) -> None:
    dataset = load_uci_har_split(tiny_uci_har, "train")
    train_dataset, validation_dataset = split_dataset_by_subject(
        dataset, validation_subjects=[2]
    )

    assert set(train_dataset.subjects.tolist()) == {1, 3}
    assert set(validation_dataset.subjects.tolist()) == {2}
    assert len(train_dataset) + len(validation_dataset) == len(dataset)


def test_create_data_loaders_returns_complete_batches(tiny_uci_har: Path) -> None:
    loaders = create_data_loaders(
        tiny_uci_har,
        batch_size=2,
        validation_subjects=[2],
        seed=7,
    )

    assert set(loaders) == {"train", "val", "test"}
    signals, labels, subjects = next(iter(loaders["val"]))
    assert signals.shape == (2, 9, 128)
    assert labels.tolist() == [2, 3]
    assert subjects.tolist() == [2, 2]


def test_iter_signal_dicts_uses_schema_order() -> None:
    signals = torch.arange(9, dtype=torch.float32)[:, None].expand(9, 128)
    window = next(iter_signal_dicts(signals))

    assert tuple(window) == SIGNAL_CHANNELS
    for channel_index, channel in enumerate(SIGNAL_CHANNELS):
        assert np.all(window[channel] == channel_index)


def test_dataset_rejects_invalid_shape_and_loader_rejects_misalignment(
    tiny_uci_har: Path,
) -> None:
    with pytest.raises(ValueError, match="形状必须"):
        UCIHARDataset(np.zeros((2, 8, 128)), np.zeros(2), np.ones(2))

    # 删除一个通道的一行，验证跨文件样本对齐检查确实生效。
    mismatched_file = (
        tiny_uci_har / "train" / "Inertial Signals" / "body_acc_x_train.txt"
    )
    values = np.loadtxt(mismatched_file)[:-1]
    np.savetxt(mismatched_file, values)
    with pytest.raises(ValueError, match="样本数不一致"):
        load_uci_har_split(tiny_uci_har, "train")
