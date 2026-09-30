"""39 维特征预处理、标准化和持久化测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_ref import FEATURE_DIM, FEATURE_NAMES, extract_features
from training.data import SIGNAL_CHANNELS, UCIHARDataset
from training.preprocess import (
    METADATA_FILENAME,
    SPLIT_FILENAMES,
    STANDARDIZATION_FILENAME,
    ProcessedFeatureDataset,
    StandardizationParameters,
    extract_dataset_features,
    load_processed_split,
    load_standardization_parameters,
    preprocess_uci_har,
)


def _make_raw_dataset() -> UCIHARDataset:
    """构造含三个受试者、且每个窗口统计量不同的原始数据集。"""
    time = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    windows = []
    for sample_index in range(6):
        channels = []
        for channel_index in range(len(SIGNAL_CHANNELS)):
            channels.append(
                time * (channel_index + 1) * (sample_index + 1) * 0.01
                + sample_index * 0.1
                + channel_index * 0.01
            )
        windows.append(channels)
    return UCIHARDataset(
        np.asarray(windows, dtype=np.float32),
        np.arange(6, dtype=np.int64),
        np.repeat(np.arange(1, 4, dtype=np.int64), 2),
    )


def _write_split(root: Path, split: str, labels: list[int], subjects: list[int]) -> None:
    """写入一个小型但目录结构完整的 UCI HAR 分区。"""
    signal_dir = root / split / "Inertial Signals"
    signal_dir.mkdir(parents=True)
    time = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    for channel_index, channel in enumerate(SIGNAL_CHANNELS):
        rows = np.stack(
            [
                time * (channel_index + 1) * (row_index + 1) * 0.01
                + row_index * 0.1
                + channel_index * 0.01
                for row_index in range(len(labels))
            ]
        )
        np.savetxt(signal_dir / f"{channel}_{split}.txt", rows)
    np.savetxt(root / split / f"y_{split}.txt", labels, fmt="%d")
    np.savetxt(root / split / f"subject_{split}.txt", subjects, fmt="%d")


@pytest.fixture
def tiny_uci_har_for_preprocessing(tmp_path: Path) -> Path:
    root = tmp_path / "UCI HAR Dataset"
    _write_split(root, "train", [1, 2, 3, 4, 5, 6], [1, 1, 2, 2, 3, 3])
    _write_split(root, "test", [6, 1], [4, 4])
    return root


def test_extract_dataset_features_matches_single_window_extractor() -> None:
    dataset = _make_raw_dataset()

    features = extract_dataset_features(dataset, batch_size=2)
    expected_first = extract_features(
        {
            channel: dataset.signals[0, index].numpy()
            for index, channel in enumerate(SIGNAL_CHANNELS)
        }
    )

    assert features.shape == (6, FEATURE_DIM)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features[0], expected_first)


def test_standardization_uses_population_statistics_and_handles_constants() -> None:
    features = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    features[:, 0] = [1.0, 2.0, 3.0]
    parameters = StandardizationParameters.fit(features)

    transformed = parameters.transform(features)

    assert parameters.mean[0] == pytest.approx(2.0)
    assert parameters.scale[0] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=0))
    assert np.all(parameters.scale[1:] == 1.0)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(parameters.inverse_transform(transformed), features, atol=1e-6)


def test_processed_split_and_parameters_round_trip(tmp_path: Path) -> None:
    features = np.arange(2 * FEATURE_DIM, dtype=np.float32).reshape(2, FEATURE_DIM)
    dataset = ProcessedFeatureDataset(features, np.array([0, 5]), np.array([1, 2]))
    parameters = StandardizationParameters.fit(features)

    from training.preprocess import save_processed_split, save_standardization_parameters

    split_path = save_processed_split(dataset, tmp_path / "train.npz")
    parameter_path = save_standardization_parameters(parameters, tmp_path / "params.npz")
    loaded_dataset = load_processed_split(split_path)
    loaded_parameters = load_standardization_parameters(parameter_path)

    assert torch.equal(loaded_dataset.features, dataset.features)
    assert torch.equal(loaded_dataset.labels, dataset.labels)
    np.testing.assert_array_equal(loaded_parameters.mean, parameters.mean)
    np.testing.assert_array_equal(loaded_parameters.scale, parameters.scale)


def test_preprocess_uci_har_saves_all_outputs_without_leakage(
    tiny_uci_har_for_preprocessing: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "processed"
    result = preprocess_uci_har(
        tiny_uci_har_for_preprocessing,
        output_dir,
        validation_subjects=[2],
        extraction_batch_size=2,
    )

    assert all((output_dir / filename).is_file() for filename in SPLIT_FILENAMES.values())
    assert (output_dir / STANDARDIZATION_FILENAME).is_file()
    assert (output_dir / METADATA_FILENAME).is_file()
    assert set(result.datasets["train"].subjects.tolist()) == {1, 3}
    assert set(result.datasets["val"].subjects.tolist()) == {2}
    assert set(result.datasets["test"].subjects.tolist()) == {4}

    # 训练特征应围绕 0；验证/测试不应被各自重新中心化，否则会造成统计量泄漏。
    np.testing.assert_allclose(
        result.datasets["train"].features.numpy().mean(axis=0), 0.0, atol=1e-5
    )
    assert not np.allclose(result.datasets["val"].features.numpy().mean(axis=0), 0.0)

    metadata = json.loads((output_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["validation_subjects"] == [2]
    assert metadata["splits"]["test"]["sample_count"] == 2


def test_preprocess_refuses_to_overwrite_existing_results(
    tiny_uci_har_for_preprocessing: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "processed"
    preprocess_uci_har(
        tiny_uci_har_for_preprocessing,
        output_dir,
        validation_subjects=[2],
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        preprocess_uci_har(
            tiny_uci_har_for_preprocessing,
            output_dir,
            validation_subjects=[2],
        )
