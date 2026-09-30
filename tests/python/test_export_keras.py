"""PyTorch 最佳检查点到 Keras 模型的转换测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES, TinyMLP, count_trainable_parameters
from training.data import ACTIVITY_NAMES
from training.export_keras import export_checkpoint_to_keras
from training.preprocess import (
    SPLIT_FILENAMES,
    ProcessedFeatureDataset,
    save_processed_split,
)

tf = pytest.importorskip("tensorflow")


def _write_conversion_inputs(root: Path) -> tuple[Path, Path, np.ndarray]:
    processed_dir = root / "processed"
    checkpoint_path = root / "best_model.pt"
    processed_dir.mkdir(parents=True)

    generator = np.random.default_rng(123)
    features = generator.normal(size=(17, FEATURE_DIM)).astype(np.float32)
    labels = generator.integers(0, NUM_CLASSES, size=17, dtype=np.int64)
    subjects = np.repeat(np.arange(1, 18, dtype=np.int64), 1)
    save_processed_split(
        ProcessedFeatureDataset(features, labels, subjects),
        processed_dir / SPLIT_FILENAMES["test"],
    )

    torch.manual_seed(456)
    model = TinyMLP()
    torch.save(
        {
            "format_version": 1,
            "epoch": 7,
            "model_name": "TinyMLP",
            "model_config": {
                "input_dim": FEATURE_DIM,
                "hidden_dims": HIDDEN_DIMS,
                "num_classes": NUM_CLASSES,
                "parameter_count": count_trainable_parameters(model),
            },
            "model_state_dict": model.state_dict(),
            "performance": {"val_accuracy": 0.75},
            "feature_names": FEATURE_NAMES,
            "activity_names": ACTIVITY_NAMES,
            "standardization": {
                "mean": torch.zeros(FEATURE_DIM, dtype=torch.float64),
                "scale": torch.ones(FEATURE_DIM, dtype=torch.float64),
                "minimum_scale": 1e-8,
            },
        },
        checkpoint_path,
    )
    return processed_dir, checkpoint_path, features


def test_export_checkpoint_to_keras_preserves_weights_and_logits(tmp_path: Path) -> None:
    processed_dir, checkpoint_path, features = _write_conversion_inputs(tmp_path)
    model_path = tmp_path / "exported" / "best_model.keras"
    report_path = tmp_path / "exported" / "report.json"

    result = export_checkpoint_to_keras(
        checkpoint_path,
        processed_dir,
        model_path,
        report_path,
    )

    assert result.model_path.is_file()
    assert result.report_path.is_file()
    assert result.sample_count == features.shape[0]
    assert result.prediction_agreement == 1.0
    assert result.max_absolute_error < 1e-5

    pytorch_model = TinyMLP()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pytorch_model.load_state_dict(checkpoint["model_state_dict"])
    pytorch_model.eval()
    keras_model = tf.keras.models.load_model(model_path, compile=False)
    with torch.inference_mode():
        expected_logits = pytorch_model(torch.from_numpy(features)).numpy()
    actual_logits = keras_model(features, training=False).numpy()
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-5, atol=1e-5)

    for pytorch_index, keras_name in ((0, "dense_1"), (2, "dense_2"), (4, "logits")):
        kernel, bias = keras_model.get_layer(keras_name).get_weights()
        np.testing.assert_array_equal(
            kernel, checkpoint["model_state_dict"][f"network.{pytorch_index}.weight"].numpy().T
        )
        np.testing.assert_array_equal(
            bias, checkpoint["model_state_dict"][f"network.{pytorch_index}.bias"].numpy()
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model"]["layers"] == ["Dense", "ReLU", "Dense", "ReLU", "Dense"]
    assert report["model"]["output"] == "logits"
    assert report["verification"]["after_reload"]["prediction_agreement"] == 1.0
    assert report["keras_model_sha256"] == result.model_sha256


def test_export_checkpoint_to_keras_rejects_existing_outputs(tmp_path: Path) -> None:
    processed_dir, checkpoint_path, _features = _write_conversion_inputs(tmp_path)
    model_path = tmp_path / "best_model.keras"
    report_path = tmp_path / "report.json"
    export_checkpoint_to_keras(
        checkpoint_path,
        processed_dir,
        model_path,
        report_path,
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        export_checkpoint_to_keras(
            checkpoint_path,
            processed_dir,
            model_path,
            report_path,
        )
