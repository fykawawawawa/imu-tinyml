"""使用训练集代表数据将 Keras TinyMLP 转换为全 INT8 TFLite 模型。

TensorFlow Lite Converter 不能直接把已有 ``.tflite`` FlatBuffer 作为再次转换的输入，
因此量化仍从已经对齐的 Keras 桥接模型开始；现有 FP32 TFLite 模型作为数值和分类基线。

默认 ``raw`` 校准策略使用全部标准化训练特征。``clip`` 策略允许指定统一的输入裁剪
上下界，并沿用相同的转换、验证和报告流程。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES
from training.data import ACTIVITY_NAMES
from training.export_keras import _import_tensorflow
from training.export_tflite import _validate_keras_model_contract
from training.preprocess import SPLIT_FILENAMES, load_processed_split

if TYPE_CHECKING:
    from keras import Model

CalibrationMode = Literal["raw", "clip"]

INT8_EXPORT_FORMAT_VERSION = 1
DEFAULT_KERAS_MODEL_PATH = Path("models/exported/best_model.keras")
DEFAULT_FLOAT_TFLITE_MODEL_PATH = Path("models/exported/best_model_float32.tflite")
DEFAULT_INT8_TFLITE_MODEL_PATH = Path(
    "models/exported/best_model_int8_baseline.tflite"
)
DEFAULT_INT8_TFLITE_REPORT_PATH = Path(
    "models/exported/best_model_int8_baseline_report.json"
)


@dataclass(frozen=True)
class InputCalibrationPolicy:
    """代表性输入在送入校准器前采用的确定性变换。

    ``raw`` 不改变已经标准化的 39 维特征；``clip`` 在标准化之后、量化之前按统一上下界
    裁剪。统一边界与 TFLite 输入张量共享单一 scale/zero-point 的契约相匹配。
    """

    mode: CalibrationMode = "raw"
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self) -> None:
        if self.mode == "raw":
            if self.clip_min is not None or self.clip_max is not None:
                raise ValueError("raw 校准策略不能设置裁剪上下界")
            return
        if self.mode != "clip":
            raise ValueError(f"不支持的校准策略：{self.mode!r}")
        if self.clip_min is None or self.clip_max is None:
            raise ValueError("clip 校准策略必须同时设置 clip_min 和 clip_max")
        if not np.isfinite(self.clip_min) or not np.isfinite(self.clip_max):
            raise ValueError("裁剪上下界必须是有限数")
        if self.clip_min >= self.clip_max:
            raise ValueError("clip_min 必须小于 clip_max")

    def apply(self, features: np.ndarray) -> np.ndarray:
        """按策略返回连续 ``float32`` 特征，不修改调用方数组。"""
        values = np.ascontiguousarray(features, dtype=np.float32)
        if self.mode == "raw":
            return values.copy()
        return np.ascontiguousarray(
            np.clip(values, self.clip_min, self.clip_max), dtype=np.float32
        )

    def as_report_dict(self) -> dict[str, object]:
        """返回可直接写入 JSON 的策略描述。"""
        return {
            "mode": self.mode,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }


@dataclass(frozen=True)
class Int8TFLiteExportResult:
    """全 INT8 TFLite 产物及验证摘要。"""

    model_path: Path
    report_path: Path
    calibration_mode: CalibrationMode
    calibration_sample_count: int
    validation_sample_count: int
    prediction_agreement: float
    float_accuracy: float
    int8_accuracy: float
    float_macro_f1: float
    int8_macro_f1: float
    model_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _feature_statistics(
    original: np.ndarray,
    transformed: np.ndarray,
) -> dict[str, object]:
    """记录校准策略前后的范围及实际裁剪规模。"""
    changed = original != transformed
    return {
        "sample_count": int(original.shape[0]),
        "scalar_count": int(original.size),
        "original": {
            "minimum": float(np.min(original)),
            "maximum": float(np.max(original)),
            "percentile_0_1": float(np.percentile(original, 0.1)),
            "percentile_99_9": float(np.percentile(original, 99.9)),
        },
        "transformed": {
            "minimum": float(np.min(transformed)),
            "maximum": float(np.max(transformed)),
            "percentile_0_1": float(np.percentile(transformed, 0.1)),
            "percentile_99_9": float(np.percentile(transformed, 99.9)),
        },
        "changed_scalar_count": int(np.count_nonzero(changed)),
        "changed_sample_count": int(np.count_nonzero(np.any(changed, axis=1))),
    }


def _representative_dataset(features: np.ndarray):
    """构造逐样本校准生成器工厂，满足 Converter 的可重复迭代接口。"""

    def generate():
        # 校准器要求保留批量维，因此每次产生形状为 (1, 39) 的 float32 输入。
        for sample in features:
            yield [sample[np.newaxis, :]]

    return generate


def _create_interpreter(model_path: Path):
    """创建无默认 delegate、单线程的解释器，保证验证结果可复现。"""
    tf = _import_tensorflow()
    return tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )


def _run_float_tflite(model_path: Path, features: np.ndarray) -> np.ndarray:
    """运行 FP32 TFLite 基线并返回 logits。"""
    values = np.ascontiguousarray(features, dtype=np.float32)
    interpreter = _create_interpreter(model_path)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.float32 or output_detail["dtype"] != np.float32:
        raise ValueError("浮点基线模型的输入输出必须都是 float32")
    interpreter.resize_tensor_input(
        int(input_detail["index"]), [values.shape[0], FEATURE_DIM], strict=True
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(int(input_detail["index"]), values)
    interpreter.invoke()
    return interpreter.get_tensor(int(output_detail["index"]))


def _quantization_parameters(detail: dict[str, object]) -> tuple[float, int]:
    """读取并校验单尺度张量的 scale 和 zero-point。"""
    scale, zero_point = detail["quantization"]
    scale = float(scale)
    zero_point = int(zero_point)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"张量 {detail['name']} 缺少有效量化 scale")
    if not -128 <= zero_point <= 127:
        raise RuntimeError(f"张量 {detail['name']} 的 zero-point 超出 INT8 范围")
    return scale, zero_point


def _run_int8_tflite(
    model_path: Path,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """量化输入、执行 INT8 推理，并返回整数与反量化 logits。"""
    values = np.ascontiguousarray(features, dtype=np.float32)
    interpreter = _create_interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError("INT8 模型必须恰好包含一个输入和一个输出")
    input_detail = input_details[0]
    output_detail = output_details[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError(
            "全整数量化模型的输入输出必须为 int8，实际为 "
            f"{input_detail['dtype']}/{output_detail['dtype']}"
        )
    input_signature = tuple(int(item) for item in input_detail["shape_signature"])
    output_signature = tuple(int(item) for item in output_detail["shape_signature"])
    if input_signature != (-1, FEATURE_DIM) or output_signature != (-1, NUM_CLASSES):
        raise RuntimeError(
            f"INT8 模型动态形状不正确：{input_signature}/{output_signature}"
        )

    input_scale, input_zero_point = _quantization_parameters(input_detail)
    output_scale, output_zero_point = _quantization_parameters(output_detail)
    unbounded_input = np.rint(values / input_scale) + input_zero_point
    saturation_mask = (unbounded_input < -128) | (unbounded_input > 127)
    quantized_input = np.clip(unbounded_input, -128, 127).astype(np.int8)

    interpreter.resize_tensor_input(
        int(input_detail["index"]), [values.shape[0], FEATURE_DIM], strict=True
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(int(input_detail["index"]), quantized_input)
    interpreter.invoke()
    quantized_logits = interpreter.get_tensor(int(output_detail["index"]))
    dequantized_logits = (
        quantized_logits.astype(np.float32) - output_zero_point
    ) * output_scale

    tensor_details = interpreter.get_tensor_details()
    floating_tensors = [
        str(item["name"])
        for item in tensor_details
        if np.issubdtype(np.dtype(item["dtype"]), np.floating)
    ]
    if floating_tensors:
        raise RuntimeError(
            "INT8 模型仍含浮点运行时张量：" + ", ".join(floating_tensors)
        )
    get_ops_details = getattr(interpreter, "_get_ops_details", None)
    operators = (
        [str(item["op_name"]) for item in get_ops_details()]
        if get_ops_details is not None
        else []
    )
    audit: dict[str, object] = {
        "input": {
            "name": str(input_detail["name"]),
            "dtype": "int8",
            "shape_signature": list(input_signature),
            "scale": input_scale,
            "zero_point": input_zero_point,
        },
        "output": {
            "name": str(output_detail["name"]),
            "dtype": "int8",
            "shape_signature": list(output_signature),
            "scale": output_scale,
            "zero_point": output_zero_point,
        },
        "operators": operators,
        "runtime_tensor_dtypes": {
            dtype: int(
                sum(np.dtype(item["dtype"]).name == dtype for item in tensor_details)
            )
            for dtype in sorted({np.dtype(item["dtype"]).name for item in tensor_details})
        },
        "input_saturation": {
            "scalar_count": int(np.count_nonzero(saturation_mask)),
            "scalar_fraction": float(np.mean(saturation_mask)),
            "sample_count": int(np.count_nonzero(np.any(saturation_mask, axis=1))),
        },
        "output_endpoint_count": {
            "negative_128": int(np.count_nonzero(quantized_logits == -128)),
            "positive_127": int(np.count_nonzero(quantized_logits == 127)),
        },
    }
    return quantized_logits, dequantized_logits, audit


def _classification_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    """计算总体和逐类分类指标。"""
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0, dtype=np.float64)
    actual_count = confusion.sum(axis=1, dtype=np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        actual_count,
        out=np.zeros_like(true_positive),
        where=actual_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "label": index,
                "activity": ACTIVITY_NAMES[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(actual_count[index]),
            }
            for index in range(NUM_CLASSES)
        ],
        "confusion_matrix": confusion.tolist(),
    }


def export_keras_to_int8_tflite(
    keras_model_path: str | Path,
    float_tflite_model_path: str | Path,
    processed_dir: str | Path,
    model_path: str | Path = DEFAULT_INT8_TFLITE_MODEL_PATH,
    report_path: str | Path = DEFAULT_INT8_TFLITE_REPORT_PATH,
    *,
    calibration_policy: InputCalibrationPolicy | None = None,
    overwrite: bool = False,
) -> Int8TFLiteExportResult:
    """使用训练分区校准并生成经过验证的全 INT8 TFLite 模型。"""
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值")
    policy = InputCalibrationPolicy() if calibration_policy is None else calibration_policy
    if not isinstance(policy, InputCalibrationPolicy):
        raise TypeError("calibration_policy 必须是 InputCalibrationPolicy 或 None")

    keras_source = Path(keras_model_path).expanduser()
    float_tflite_source = Path(float_tflite_model_path).expanduser()
    processed_source = Path(processed_dir).expanduser()
    destination = Path(model_path).expanduser()
    report_destination = Path(report_path).expanduser()
    for source, description in (
        (keras_source, "Keras 桥接模型"),
        (float_tflite_source, "浮点 TFLite 基线"),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"找不到{description}：{source}")
    if destination.suffix != ".tflite":
        raise ValueError(f"INT8 模型路径必须使用 .tflite 后缀：{destination}")
    existing = [path for path in (destination, report_destination) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"量化输出已存在；如需覆盖请设置 overwrite=True：{joined}")

    train_dataset = load_processed_split(processed_source / SPLIT_FILENAMES["train"])
    validation_dataset = load_processed_split(
        processed_source / SPLIT_FILENAMES["val"]
    )
    calibration_original = train_dataset.features.numpy()
    calibration_features = policy.apply(calibration_original)
    validation_original = validation_dataset.features.numpy()
    validation_features = policy.apply(validation_original)

    tf = _import_tensorflow()
    keras_model: Model = tf.keras.models.load_model(keras_source, compile=False)
    _validate_keras_model_contract(keras_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset(calibration_features)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_payload = converter.convert()
    if not isinstance(tflite_payload, bytes) or not tflite_payload:
        raise RuntimeError("TensorFlow Lite Converter 未返回有效 INT8 FlatBuffer")

    destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    verification_path = destination.with_name(f".{destination.name}.verify")
    verification_path.unlink(missing_ok=True)
    try:
        _atomic_write_bytes(verification_path, tflite_payload)
        float_logits = _run_float_tflite(float_tflite_source, validation_features)
        quantized_logits, dequantized_logits, tensor_audit = _run_int8_tflite(
            verification_path, validation_features
        )
        float_predictions = np.argmax(float_logits, axis=1)
        int8_predictions = np.argmax(quantized_logits, axis=1)
        labels = validation_dataset.labels.numpy()
        float_metrics = _classification_summary(labels, float_predictions)
        int8_metrics = _classification_summary(labels, int8_predictions)
        prediction_agreement = float(np.mean(float_predictions == int8_predictions))
        logit_error = np.abs(float_logits - dequantized_logits)
        os.replace(verification_path, destination)
    finally:
        verification_path.unlink(missing_ok=True)

    model_sha256 = _sha256(destination)
    result = Int8TFLiteExportResult(
        model_path=destination,
        report_path=report_destination,
        calibration_mode=policy.mode,
        calibration_sample_count=int(calibration_features.shape[0]),
        validation_sample_count=int(validation_features.shape[0]),
        prediction_agreement=prediction_agreement,
        float_accuracy=float(float_metrics["accuracy"]),
        int8_accuracy=float(int8_metrics["accuracy"]),
        float_macro_f1=float(float_metrics["macro_f1"]),
        int8_macro_f1=float(int8_metrics["macro_f1"]),
        model_sha256=model_sha256,
    )
    report: dict[str, object] = {
        "format_version": INT8_EXPORT_FORMAT_VERSION,
        "conversion": "keras_float32_to_tflite_full_int8",
        "source_keras_model": str(keras_source.resolve()),
        "source_keras_sha256": _sha256(keras_source),
        "reference_float_tflite": str(float_tflite_source.resolve()),
        "reference_float_tflite_sha256": _sha256(float_tflite_source),
        "int8_tflite_model": str(destination.resolve()),
        "int8_tflite_sha256": model_sha256,
        "model": {
            "structure": [FEATURE_DIM, *HIDDEN_DIMS, NUM_CLASSES],
            "parameter_count": int(keras_model.count_params()),
            "numeric_format": "full_int8",
            "output": "logits",
        },
        "feature_names": list(FEATURE_NAMES),
        "activity_names": list(ACTIVITY_NAMES),
        "calibration": {
            "split": "train",
            "uses_all_samples": True,
            "policy": policy.as_report_dict(),
            "statistics": _feature_statistics(
                calibration_original, calibration_features
            ),
        },
        "validation": {
            "split": "val",
            "sample_count": int(validation_features.shape[0]),
            "policy_statistics": _feature_statistics(
                validation_original, validation_features
            ),
            "float_metrics": float_metrics,
            "int8_metrics": int8_metrics,
            "accuracy_delta": result.int8_accuracy - result.float_accuracy,
            "macro_f1_delta": result.int8_macro_f1 - result.float_macro_f1,
            "prediction_agreement": prediction_agreement,
            "dequantized_logit_error": {
                "maximum_absolute": float(np.max(logit_error)),
                "mean_absolute": float(np.mean(logit_error)),
            },
        },
        "tensor_audit": tensor_audit,
        "versions": {
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
        },
        "result": {
            "model_path": str(destination.resolve()),
            "report_path": str(report_destination.resolve()),
            "calibration_mode": result.calibration_mode,
            "calibration_sample_count": result.calibration_sample_count,
            "validation_sample_count": result.validation_sample_count,
            "prediction_agreement": result.prediction_agreement,
            "float_accuracy": result.float_accuracy,
            "int8_accuracy": result.int8_accuracy,
            "float_macro_f1": result.float_macro_f1,
            "int8_macro_f1": result.int8_macro_f1,
            "model_sha256": result.model_sha256,
        },
    }
    _atomic_save_json(report_destination, report)
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras-model", type=Path, default=DEFAULT_KERAS_MODEL_PATH)
    parser.add_argument(
        "--float-tflite", type=Path, default=DEFAULT_FLOAT_TFLITE_MODEL_PATH
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=DEFAULT_INT8_TFLITE_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_INT8_TFLITE_REPORT_PATH)
    parser.add_argument("--clip-min", type=float)
    parser.add_argument("--clip-max", type=float)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有量化输出")
    return parser


def _policy_from_arguments(arguments: argparse.Namespace) -> InputCalibrationPolicy:
    """从成对裁剪参数构造策略；两者都省略时使用原始输入。"""
    if arguments.clip_min is None and arguments.clip_max is None:
        return InputCalibrationPolicy()
    return InputCalibrationPolicy(
        mode="clip",
        clip_min=arguments.clip_min,
        clip_max=arguments.clip_max,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    result = export_keras_to_int8_tflite(
        arguments.keras_model,
        arguments.float_tflite,
        arguments.processed_dir,
        arguments.output,
        arguments.report,
        calibration_policy=_policy_from_arguments(arguments),
        overwrite=arguments.overwrite,
    )
    print(f"INT8 TFLite 模型：{result.model_path.resolve()}")
    print(f"量化报告：{result.report_path.resolve()}")
    print(
        f"校准策略={result.calibration_mode}，"
        f"训练样本={result.calibration_sample_count}，"
        f"验证预测一致率={result.prediction_agreement:.2%}"
    )
    print(
        f"验证 accuracy：FP32={result.float_accuracy:.4%}，"
        f"INT8={result.int8_accuracy:.4%}；"
        f"macro-F1：FP32={result.float_macro_f1:.4f}，"
        f"INT8={result.int8_macro_f1:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FLOAT_TFLITE_MODEL_PATH",
    "DEFAULT_INT8_TFLITE_MODEL_PATH",
    "DEFAULT_INT8_TFLITE_REPORT_PATH",
    "DEFAULT_KERAS_MODEL_PATH",
    "INT8_EXPORT_FORMAT_VERSION",
    "InputCalibrationPolicy",
    "Int8TFLiteExportResult",
    "export_keras_to_int8_tflite",
    "main",
]
