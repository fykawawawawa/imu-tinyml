"""最佳模型测试集评估与报告产物测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES, TinyMLP, count_trainable_parameters
from training.data import ACTIVITY_NAMES
from training.evaluate import (
    EvaluationConfig,
    evaluate_best_model,
    metrics_from_confusion_matrix,
)
from training.preprocess import (
    SPLIT_FILENAMES,
    STANDARDIZATION_FILENAME,
    ProcessedFeatureDataset,
    StandardizationParameters,
    save_processed_split,
    save_standardization_parameters,
)


def _write_evaluation_inputs(
    root: Path,
    *,
    invalid_feature_names: bool = False,
) -> tuple[Path, Path]:
    """生成预测恒为第 0 类的确定性检查点和测试集。"""
    processed_dir = root / "processed"
    checkpoint_path = root / "best_model.pt"
    processed_dir.mkdir(parents=True)

    features = np.zeros((8, FEATURE_DIM), dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 2, 3, 5], dtype=np.int64)
    subjects = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    save_processed_split(
        ProcessedFeatureDataset(features, labels, subjects),
        processed_dir / SPLIT_FILENAMES["test"],
    )
    parameters = StandardizationParameters(
        mean=np.zeros(FEATURE_DIM), scale=np.ones(FEATURE_DIM)
    )
    save_standardization_parameters(
        parameters, processed_dir / STANDARDIZATION_FILENAME
    )

    model = TinyMLP()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias[0] = 1.0
    feature_names = list(FEATURE_NAMES)
    if invalid_feature_names:
        feature_names[0] = "wrong_feature"
    torch.save(
        {
            "format_version": 1,
            "epoch": 3,
            "model_name": "TinyMLP",
            "model_config": {
                "input_dim": FEATURE_DIM,
                "hidden_dims": HIDDEN_DIMS,
                "num_classes": NUM_CLASSES,
                "parameter_count": count_trainable_parameters(model),
            },
            "model_state_dict": model.state_dict(),
            "performance": {"val_accuracy": 0.5, "val_macro_f1": 0.4},
            "feature_names": feature_names,
            "activity_names": ACTIVITY_NAMES,
            "standardization": {
                "mean": torch.zeros(FEATURE_DIM, dtype=torch.float64),
                "scale": torch.ones(FEATURE_DIM, dtype=torch.float64),
                "minimum_scale": 1e-8,
            },
        },
        checkpoint_path,
    )
    return processed_dir, checkpoint_path


def test_metrics_from_confusion_matrix_computes_per_class_and_weighted_values() -> None:
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    confusion[0, 0] = 3
    confusion[1, 0] = 2
    confusion[2, 2] = 1

    metrics = metrics_from_confusion_matrix(confusion, loss=0.25)

    assert metrics.sample_count == 6
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.class_metrics[0].precision == pytest.approx(3 / 5)
    assert metrics.class_metrics[0].recall == pytest.approx(1.0)
    assert metrics.class_metrics[1].f1 == 0.0
    assert metrics.class_metrics[2].f1 == 1.0
    assert metrics.weighted_recall == pytest.approx(metrics.accuracy)


def test_evaluate_best_model_generates_complete_artifacts(tmp_path: Path) -> None:
    processed_dir, checkpoint_path = _write_evaluation_inputs(tmp_path)
    output_dir = tmp_path / "evaluation"

    result = evaluate_best_model(
        checkpoint_path,
        processed_dir,
        output_dir,
        config=EvaluationConfig(batch_size=3, device="cpu"),
    )

    assert result.metrics.sample_count == 8
    assert result.metrics.accuracy == pytest.approx(3 / 8)
    assert result.confusion_matrix.sum().item() == 8
    assert result.confusion_matrix[0, 0].item() == 3
    assert result.confusion_matrix[:, 0].sum().item() == 8
    for path in (
        result.metrics_path,
        result.report_path,
        result.confusion_csv_path,
        result.confusion_png_path,
        result.log_path,
    ):
        assert path.is_file()
        assert path.stat().st_size > 0

    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["model"]["feature_dim"] == FEATURE_DIM
    assert payload["evaluation_split"] == "official_test"
    assert len(payload["test"]["per_class"]) == NUM_CLASSES
    assert payload["confusion_matrix_orientation"]["rows"] == "true_activity"

    with result.report_path.open("r", encoding="utf-8-sig", newline="") as stream:
        report_rows = list(csv.reader(stream))
    assert report_rows[0] == ["label", "activity", "precision", "recall", "f1", "support"]
    assert report_rows[1][1] == "WALKING"
    assert report_rows[-1][1] == "ACCURACY"

    log_text = result.log_path.read_text(encoding="utf-8")
    assert "官方测试集评估" in log_text
    assert f"特征维度={FEATURE_DIM}" in log_text
    assert "macro_f1=" in log_text


def test_evaluate_rejects_incompatible_checkpoint_and_existing_outputs(
    tmp_path: Path,
) -> None:
    invalid_root = tmp_path / "invalid"
    processed_dir, invalid_checkpoint = _write_evaluation_inputs(
        invalid_root, invalid_feature_names=True
    )
    with pytest.raises(ValueError, match="特征名称或顺序"):
        evaluate_best_model(
            invalid_checkpoint,
            processed_dir,
            tmp_path / "invalid-output",
            config=EvaluationConfig(device="cpu"),
        )

    valid_root = tmp_path / "valid"
    processed_dir, checkpoint_path = _write_evaluation_inputs(valid_root)
    output_dir = tmp_path / "valid-output"
    evaluate_best_model(
        checkpoint_path,
        processed_dir,
        output_dir,
        config=EvaluationConfig(device="cpu"),
    )
    with pytest.raises(FileExistsError, match="overwrite=True"):
        evaluate_best_model(
            checkpoint_path,
            processed_dir,
            output_dir,
            config=EvaluationConfig(device="cpu"),
        )
