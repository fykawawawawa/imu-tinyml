"""Keras 桥接模型到浮点 TFLite 的转换测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_ref import FEATURE_DIM
from models.mlp import NUM_CLASSES
from training.export_keras import build_keras_tiny_mlp
from training.export_tflite import export_keras_to_float_tflite
from training.preprocess import (
    SPLIT_FILENAMES,
    ProcessedFeatureDataset,
    save_processed_split,
)

tf = pytest.importorskip("tensorflow")


def _write_float_tflite_inputs(root: Path) -> tuple[Path, Path, np.ndarray]:
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True)
    keras_model_path = root / "best_model.keras"

    tf.keras.utils.set_random_seed(2468)
    keras_model = build_keras_tiny_mlp()
    keras_model.save(keras_model_path, include_optimizer=False)

    generator = np.random.default_rng(1357)
    features = generator.normal(size=(19, FEATURE_DIM)).astype(np.float32)
    labels = generator.integers(0, NUM_CLASSES, size=19, dtype=np.int64)
    subjects = np.arange(1, 20, dtype=np.int64)
    save_processed_split(
        ProcessedFeatureDataset(features, labels, subjects),
        processed_dir / SPLIT_FILENAMES["test"],
    )
    return processed_dir, keras_model_path, features


def test_export_keras_to_float_tflite_preserves_logits(tmp_path: Path) -> None:
    processed_dir, keras_model_path, features = _write_float_tflite_inputs(tmp_path)
    tflite_path = tmp_path / "exported" / "best_model_float32.tflite"
    report_path = tmp_path / "exported" / "report.json"

    result = export_keras_to_float_tflite(
        keras_model_path,
        processed_dir,
        tflite_path,
        report_path,
    )

    assert result.model_path.is_file()
    assert result.report_path.is_file()
    assert result.sample_count == features.shape[0]
    assert result.prediction_agreement == 1.0
    assert result.max_absolute_error < 1e-5

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    assert input_detail["dtype"] == np.float32
    assert output_detail["dtype"] == np.float32
    assert tuple(input_detail["shape_signature"]) == (-1, FEATURE_DIM)
    assert tuple(output_detail["shape_signature"]) == (-1, NUM_CLASSES)

    keras_model = tf.keras.models.load_model(keras_model_path, compile=False)
    expected_logits = keras_model(features, training=False).numpy()
    interpreter.resize_tensor_input(input_detail["index"], features.shape, strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], features)
    interpreter.invoke()
    actual_logits = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-5, atol=1e-5)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["conversion"] == "keras_float32_to_tflite_float32"
    assert report["model"]["numeric_format"] == "float32"
    assert report["tensor_contract"]["input"]["dtype"] == "float32"
    assert report["tensor_contract"]["output"]["dtype"] == "float32"
    assert report["verification"]["prediction_agreement"] == 1.0
    assert report["tflite_model_sha256"] == result.model_sha256


def test_export_keras_to_float_tflite_rejects_existing_outputs(tmp_path: Path) -> None:
    processed_dir, keras_model_path, _features = _write_float_tflite_inputs(tmp_path)
    tflite_path = tmp_path / "best_model_float32.tflite"
    report_path = tmp_path / "report.json"
    export_keras_to_float_tflite(
        keras_model_path,
        processed_dir,
        tflite_path,
        report_path,
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        export_keras_to_float_tflite(
            keras_model_path,
            processed_dir,
            tflite_path,
            report_path,
        )
