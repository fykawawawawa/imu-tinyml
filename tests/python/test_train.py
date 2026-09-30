"""轻量 MLP 训练、验证、早停、日志和检查点测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES
from training.data import ACTIVITY_NAMES
from training.preprocess import (
    SPLIT_FILENAMES,
    STANDARDIZATION_FILENAME,
    ProcessedFeatureDataset,
    StandardizationParameters,
    save_processed_split,
    save_standardization_parameters,
)
from training.train import (
    TrainingConfig,
    classification_metrics_from_confusion,
    load_training_config,
    train_model,
)


def _write_processed_data(root: Path) -> None:
    """生成一个可快速学习的六分类 39 维小数据集。"""
    generator = np.random.default_rng(123)

    def make_split(samples_per_class: int, subject_offset: int) -> ProcessedFeatureDataset:
        features: list[np.ndarray] = []
        labels: list[int] = []
        subjects: list[int] = []
        for label in range(NUM_CLASSES):
            # 每类主要由自己的一个特征维度表示，并叠加很小噪声。
            class_features = generator.normal(
                0.0, 0.05, size=(samples_per_class, FEATURE_DIM)
            ).astype(np.float32)
            class_features[:, label] += 3.0
            features.append(class_features)
            labels.extend([label] * samples_per_class)
            subjects.extend([subject_offset + label] * samples_per_class)
        return ProcessedFeatureDataset(
            np.concatenate(features),
            np.asarray(labels),
            np.asarray(subjects),
        )

    root.mkdir(parents=True)
    train_dataset = make_split(samples_per_class=8, subject_offset=1)
    val_dataset = make_split(samples_per_class=3, subject_offset=20)
    save_processed_split(train_dataset, root / SPLIT_FILENAMES["train"])
    save_processed_split(val_dataset, root / SPLIT_FILENAMES["val"])
    save_standardization_parameters(
        StandardizationParameters(
            mean=np.zeros(FEATURE_DIM),
            scale=np.ones(FEATURE_DIM),
        ),
        root / STANDARDIZATION_FILENAME,
    )


def test_classification_metrics_from_confusion() -> None:
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    confusion[0, 0] = 3
    confusion[1, 1] = 1
    confusion[1, 0] = 1

    metrics = classification_metrics_from_confusion(confusion)

    assert metrics.accuracy == pytest.approx(0.8)
    # 宏平均覆盖全部六类；没有样本的四类按 0 计入。
    assert metrics.macro_precision == pytest.approx((0.75 + 1.0) / 6.0)
    assert metrics.macro_recall == pytest.approx((1.0 + 0.5) / 6.0)


def test_load_training_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """\
epochs: 25
batch_size: 32
learning_rate: 0.002
weight_decay: 0.0
patience: 6
min_delta: 0.0002
seed: 9
num_workers: 0
device: cpu
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config == TrainingConfig(
        epochs=25,
        batch_size=32,
        learning_rate=0.002,
        weight_decay=0.0,
        patience=6,
        min_delta=0.0002,
        seed=9,
        num_workers=0,
        device="cpu",
    )


def test_load_training_config_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("epoch: 10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="未知字段：epoch"):
        load_training_config(config_path)


def test_train_model_writes_descriptive_log_history_and_checkpoint(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "checkpoints"
    _write_processed_data(processed_dir)

    result = train_model(
        processed_dir,
        output_dir,
        config=TrainingConfig(
            epochs=4,
            batch_size=12,
            learning_rate=0.02,
            weight_decay=0.0,
            patience=4,
            seed=7,
            device="cpu",
        ),
    )

    assert result.checkpoint_path.is_file()
    assert result.history_path.is_file()
    assert result.log_path.is_file()
    assert result.epochs_completed == 4

    log_text = result.log_path.read_text(encoding="utf-8")
    assert f"特征维度={FEATURE_DIM}" in log_text
    assert "模型=TinyMLP" in log_text
    assert "运动状态分类配置" in log_text
    assert "val_accuracy=" in log_text
    assert "val_macro_f1=" in log_text

    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert history["feature_dim"] == FEATURE_DIM
    assert history["model"]["structure"] == [FEATURE_DIM, *HIDDEN_DIMS, NUM_CLASSES]
    assert len(history["epochs"]) == 4

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["model_name"] == "TinyMLP"
    assert checkpoint["model_config"]["input_dim"] == FEATURE_DIM
    assert tuple(checkpoint["feature_names"]) == FEATURE_NAMES
    assert tuple(checkpoint["activity_names"]) == ACTIVITY_NAMES
    assert checkpoint["standardization"]["mean"].shape == (FEATURE_DIM,)
    assert set(checkpoint["model_state_dict"]) == {
        "network.0.weight",
        "network.0.bias",
        "network.2.weight",
        "network.2.bias",
        "network.4.weight",
        "network.4.bias",
    }


def test_early_stopping_and_overwrite_protection(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "checkpoints"
    _write_processed_data(processed_dir)
    config = TrainingConfig(
        epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=2,
        # 首轮之后不可能达到如此大的最小改善量，因此确定在第 3 轮早停。
        min_delta=1e6,
        seed=11,
        device="cpu",
    )

    result = train_model(processed_dir, output_dir, config=config)

    assert result.stopped_early
    assert result.epochs_completed == 3
    assert result.best_epoch == 1
    assert "早停触发" in result.log_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        train_model(processed_dir, output_dir, config=config)
