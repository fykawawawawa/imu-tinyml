"""全 INT8 TFLite 校准、转换和验证测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_ref import FEATURE_DIM
from models.mlp import NUM_CLASSES
from training.export_int8_tflite import (
    InputCalibrationPolicy,
    export_keras_to_int8_tflite,
)
from training.export_keras import build_keras_tiny_mlp
from training.export_tflite import export_keras_to_float_tflite
from training.preprocess import (
    SPLIT_FILENAMES,
    ProcessedFeatureDataset,
    save_processed_split,
)

tf = pytest.importorskip("tensorflow")


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True)
    keras_path = root / "best_model.keras"
    float_tflite_path = root / "best_model_float32.tflite"

    tf.keras.utils.set_random_seed(8642)
    model = build_keras_tiny_mlp()
    model.save(keras_path, include_optimizer=False)

    generator = np.random.default_rng(9753)
    for split, count in (("train", 96), ("val", 31), ("test", 17)):
        features = generator.normal(size=(count, FEATURE_DIM)).astype(np.float32)
        labels = generator.integers(0, NUM_CLASSES, size=count, dtype=np.int64)
        subjects = np.arange(1, count + 1, dtype=np.int64)
        save_processed_split(
            ProcessedFeatureDataset(features, labels, subjects),
            processed_dir / SPLIT_FILENAMES[split],
        )

    export_keras_to_float_tflite(
        keras_path,
        processed_dir,
        float_tflite_path,
        root / "float_report.json",
    )
    return processed_dir, keras_path, float_tflite_path


def test_input_calibration_policy_reserves_raw_and_clip_modes() -> None:
    values = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
    raw = InputCalibrationPolicy().apply(values)
    clipped = InputCalibrationPolicy(
        mode="clip", clip_min=-1.0, clip_max=1.0
    ).apply(values)

    np.testing.assert_array_equal(raw, values)
    np.testing.assert_array_equal(clipped, np.array([[-1.0, 0.0, 1.0]]))
    assert raw is not values
    with pytest.raises(ValueError, match="同时设置"):
        InputCalibrationPolicy(mode="clip", clip_min=-1.0)


def test_export_raw_calibrated_full_int8_model(tmp_path: Path) -> None:
    processed_dir, keras_path, float_tflite_path = _write_inputs(tmp_path)
    int8_path = tmp_path / "best_model_int8_baseline.tflite"
    report_path = tmp_path / "int8_report.json"

    result = export_keras_to_int8_tflite(
        keras_path,
        float_tflite_path,
        processed_dir,
        int8_path,
        report_path,
    )

    assert result.model_path.is_file()
    assert result.report_path.is_file()
    assert result.calibration_mode == "raw"
    assert result.calibration_sample_count == 96
    assert result.validation_sample_count == 31

    interpreter = tf.lite.Interpreter(model_path=str(int8_path), num_threads=1)
    assert interpreter.get_input_details()[0]["dtype"] == np.int8
    assert interpreter.get_output_details()[0]["dtype"] == np.int8

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["calibration"]["split"] == "train"
    assert report["calibration"]["uses_all_samples"] is True
    assert report["calibration"]["policy"]["mode"] == "raw"
    assert report["calibration"]["statistics"]["changed_scalar_count"] == 0
    assert report["tensor_audit"]["input"]["dtype"] == "int8"
    assert report["tensor_audit"]["output"]["dtype"] == "int8"
    assert report["tensor_audit"]["operators"] == [
        "FULLY_CONNECTED",
        "FULLY_CONNECTED",
        "FULLY_CONNECTED",
    ]
    assert "float32" not in report["tensor_audit"]["runtime_tensor_dtypes"]
    assert report["int8_tflite_sha256"] == result.model_sha256
