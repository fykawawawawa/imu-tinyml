"""在官方测试集上评估 FP32 与全 INT8 TFLite 模型并生成报告。

默认在 ``artifacts/int8_evaluation`` 中生成与 PyTorch 评估目录相同的五类产物：

* ``metrics.json``：FP32/INT8 指标、指标差值、预测一致率和量化审计；
* ``classification_report.csv``：INT8 模型的逐类别、宏平均和加权平均指标；
* ``confusion_matrix.csv``：INT8 模型的原始混淆矩阵；
* ``confusion_matrix.png``：INT8 模型的混淆矩阵热力图；
* ``evaluation.log``：本次测试集评估的完整文本摘要。

测试集仅用于最终评估，不参与校准，也不用于选择后续裁剪范围。
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES
from training.data import ACTIVITY_NAMES
from training.evaluate import (
    EvaluationMetrics,
    _atomic_save_json,
    _close_logger,
    _configure_logger,
    _save_classification_report,
    _save_confusion_csv,
    _save_confusion_figure,
    metrics_from_confusion_matrix,
)
from training.export_int8_tflite import (
    DEFAULT_FLOAT_TFLITE_MODEL_PATH,
    DEFAULT_INT8_TFLITE_MODEL_PATH,
    _run_float_tflite,
    _run_int8_tflite,
)
from training.preprocess import SPLIT_FILENAMES, load_processed_split

INT8_EVALUATION_FORMAT_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("artifacts/int8_evaluation")
DEFAULT_METRICS_FILENAME = "metrics.json"
DEFAULT_REPORT_FILENAME = "classification_report.csv"
DEFAULT_CONFUSION_CSV_FILENAME = "confusion_matrix.csv"
DEFAULT_CONFUSION_PNG_FILENAME = "confusion_matrix.png"
DEFAULT_LOG_FILENAME = "evaluation.log"


@dataclass(frozen=True)
class Int8TFLiteEvaluationResult:
    """一次 FP32/INT8 TFLite 测试集评估及其报告路径。"""

    float_metrics: EvaluationMetrics
    int8_metrics: EvaluationMetrics
    prediction_agreement: float
    prediction_change_count: int
    metrics_path: Path
    report_path: Path
    confusion_csv_path: Path
    confusion_png_path: Path
    log_path: Path


def _sha256(path: Path) -> str:
    """分块计算模型文件哈希，便于报告精确对应到模型二进制。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """以数值稳定的 log-sum-exp 形式计算平均交叉熵。"""
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.shape != (targets.shape[0], NUM_CLASSES):
        raise ValueError(
            f"logits 形状必须为 ({targets.shape[0]}, {NUM_CLASSES})，实际为 {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("logits 包含 NaN 或 Inf")
    row_maximum = np.max(values, axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(values - row_maximum).sum(axis=1))
    log_normalizer += row_maximum[:, 0]
    losses = log_normalizer - values[np.arange(targets.shape[0]), targets]
    return float(np.mean(losses))


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    logits: np.ndarray,
) -> tuple[EvaluationMetrics, torch.Tensor]:
    """从预测结果构造混淆矩阵，并复用正式评估的指标计算规则。"""
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    confusion_tensor = torch.from_numpy(confusion)
    metrics = metrics_from_confusion_matrix(
        confusion_tensor,
        loss=_cross_entropy_loss(logits, labels),
    )
    return metrics, confusion_tensor


def _metrics_dict(metrics: EvaluationMetrics) -> dict[str, object]:
    """把统一评估指标转换为 JSON 可序列化结构。"""
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "weighted_precision": metrics.weighted_precision,
        "weighted_recall": metrics.weighted_recall,
        "weighted_f1": metrics.weighted_f1,
        "sample_count": metrics.sample_count,
        "per_class": [asdict(item) for item in metrics.class_metrics],
    }


def evaluate_int8_tflite(
    float_model_path: str | Path,
    int8_model_path: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    overwrite: bool = False,
) -> Int8TFLiteEvaluationResult:
    """在官方测试集上比较 FP32/INT8 TFLite，并保存完整评估报告。"""
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值")

    float_model = Path(float_model_path).expanduser()
    int8_model = Path(int8_model_path).expanduser()
    processed_source = Path(processed_dir).expanduser()
    destination = Path(output_dir).expanduser()
    for source, description in (
        (float_model, "FP32 TFLite 模型"),
        (int8_model, "INT8 TFLite 模型"),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"找不到{description}：{source}")

    metrics_path = destination / DEFAULT_METRICS_FILENAME
    report_path = destination / DEFAULT_REPORT_FILENAME
    confusion_csv_path = destination / DEFAULT_CONFUSION_CSV_FILENAME
    confusion_png_path = destination / DEFAULT_CONFUSION_PNG_FILENAME
    log_path = destination / DEFAULT_LOG_FILENAME
    output_paths = (
        metrics_path,
        report_path,
        confusion_csv_path,
        confusion_png_path,
        log_path,
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"评估输出已存在；如需覆盖请设置 overwrite=True：{joined}")

    test_dataset = load_processed_split(
        processed_source / SPLIT_FILENAMES["test"]
    )
    features = test_dataset.features.numpy()
    labels = test_dataset.labels.numpy()
    subjects = test_dataset.subjects.numpy()

    float_logits = _run_float_tflite(float_model, features)
    quantized_logits, int8_logits, tensor_audit = _run_int8_tflite(
        int8_model, features
    )
    float_predictions = np.argmax(float_logits, axis=1)
    int8_predictions = np.argmax(int8_logits, axis=1)
    float_metrics, _float_confusion = _classification_metrics(
        labels, float_predictions, float_logits
    )
    int8_metrics, int8_confusion = _classification_metrics(
        labels, int8_predictions, int8_logits
    )

    prediction_changes = float_predictions != int8_predictions
    float_correct = float_predictions == labels
    int8_correct = int8_predictions == labels
    changed_to_correct = int(np.count_nonzero(prediction_changes & int8_correct))
    changed_to_incorrect = int(np.count_nonzero(prediction_changes & float_correct))
    prediction_change_count = int(np.count_nonzero(prediction_changes))
    prediction_agreement = float(np.mean(~prediction_changes))
    logit_error = np.abs(int8_logits - float_logits)

    destination.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(log_path)
    try:
        logger.info(
            "官方测试集 TFLite 对比 | 样本=%d | 受试者=%d | 结构=%d -> %d -> %d -> %d",
            labels.shape[0],
            len(set(subjects.tolist())),
            FEATURE_DIM,
            HIDDEN_DIMS[0],
            HIDDEN_DIMS[1],
            NUM_CLASSES,
        )
        logger.info("FP32 模型=%s", float_model.resolve())
        logger.info("INT8 模型=%s", int8_model.resolve())
        logger.info(
            "FP32 性能 | loss=%.6f | accuracy=%.4f%% | macro_f1=%.6f",
            float_metrics.loss,
            float_metrics.accuracy * 100.0,
            float_metrics.macro_f1,
        )
        logger.info(
            "INT8 性能 | loss=%.6f | accuracy=%.4f%% | macro_f1=%.6f",
            int8_metrics.loss,
            int8_metrics.accuracy * 100.0,
            int8_metrics.macro_f1,
        )
        logger.info(
            "量化影响 | 预测一致率=%.4f%% | 变化=%d | 变正确=%d | 变错误=%d | 净增正确=%+d",
            prediction_agreement * 100.0,
            prediction_change_count,
            changed_to_correct,
            changed_to_incorrect,
            changed_to_correct - changed_to_incorrect,
        )
        logger.info(
            "logits 误差 | 平均绝对误差=%.6f | 最大绝对误差=%.6f",
            float(np.mean(logit_error)),
            float(np.max(logit_error)),
        )
        logger.info(
            "INT8 审计 | 输入饱和标量=%d | 饱和样本=%d | 输出端点 -128=%d, 127=%d",
            tensor_audit["input_saturation"]["scalar_count"],
            tensor_audit["input_saturation"]["sample_count"],
            tensor_audit["output_endpoint_count"]["negative_128"],
            tensor_audit["output_endpoint_count"]["positive_127"],
        )
        for item in int8_metrics.class_metrics:
            logger.info(
                "INT8 类别性能 | label=%d | activity=%s | precision=%.4f | "
                "recall=%.4f | f1=%.4f | support=%d",
                item.label,
                item.activity,
                item.precision,
                item.recall,
                item.f1,
                item.support,
            )

        metrics_payload: dict[str, object] = {
            "format_version": INT8_EVALUATION_FORMAT_VERSION,
            "task": "UCI HAR 六类运动状态分类",
            "evaluation_split": "official_test",
            "models": {
                "float32_tflite": {
                    "path": str(float_model.resolve()),
                    "sha256": _sha256(float_model),
                },
                "int8_tflite": {
                    "path": str(int8_model.resolve()),
                    "sha256": _sha256(int8_model),
                },
            },
            "model": {
                "name": "TinyMLP",
                "feature_dim": FEATURE_DIM,
                "structure": [FEATURE_DIM, *HIDDEN_DIMS, NUM_CLASSES],
                "output": "logits",
            },
            "feature_names": list(FEATURE_NAMES),
            "activity_names": list(ACTIVITY_NAMES),
            "test": {
                "sample_count": int(labels.shape[0]),
                "subject_count": len(set(subjects.tolist())),
                "float32": _metrics_dict(float_metrics),
                "int8": _metrics_dict(int8_metrics),
                "difference_int8_minus_float32": {
                    "accuracy": int8_metrics.accuracy - float_metrics.accuracy,
                    "macro_f1": int8_metrics.macro_f1 - float_metrics.macro_f1,
                    "loss": int8_metrics.loss - float_metrics.loss,
                },
                "prediction_comparison": {
                    "agreement": prediction_agreement,
                    "changed_count": prediction_change_count,
                    "changed_to_correct_count": changed_to_correct,
                    "changed_to_incorrect_count": changed_to_incorrect,
                    "net_correct_change": changed_to_correct - changed_to_incorrect,
                },
                "dequantized_logit_error": {
                    "mean_absolute": float(np.mean(logit_error)),
                    "maximum_absolute": float(np.max(logit_error)),
                },
            },
            "tensor_audit": tensor_audit,
            "quantized_output_observed_range": {
                "minimum": int(np.min(quantized_logits)),
                "maximum": int(np.max(quantized_logits)),
            },
            "confusion_matrix_orientation": {
                "rows": "true_activity",
                "columns": "predicted_activity",
            },
            "report_scope": {
                "classification_report": "int8",
                "confusion_matrix": "int8",
            },
            "methodology_note": (
                "官方测试集只用于最终评估；测试结果不参与 INT8 校准或裁剪范围选择。"
            ),
        }
        _atomic_save_json(metrics_path, metrics_payload)
        _save_classification_report(report_path, int8_metrics)
        _save_confusion_csv(confusion_csv_path, int8_confusion)
        _save_confusion_figure(
            confusion_png_path,
            int8_confusion,
            accuracy=int8_metrics.accuracy,
        )
        logger.info("评估产物已保存到 %s", destination.resolve())
    finally:
        _close_logger(logger)

    return Int8TFLiteEvaluationResult(
        float_metrics=float_metrics,
        int8_metrics=int8_metrics,
        prediction_agreement=prediction_agreement,
        prediction_change_count=prediction_change_count,
        metrics_path=metrics_path,
        report_path=report_path,
        confusion_csv_path=confusion_csv_path,
        confusion_png_path=confusion_png_path,
        log_path=log_path,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--float-model", type=Path, default=DEFAULT_FLOAT_TFLITE_MODEL_PATH
    )
    parser.add_argument(
        "--int8-model", type=Path, default=DEFAULT_INT8_TFLITE_MODEL_PATH
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有评估输出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    arguments = _build_argument_parser().parse_args(argv)
    result = evaluate_int8_tflite(
        arguments.float_model,
        arguments.int8_model,
        arguments.processed_dir,
        arguments.output_dir,
        overwrite=arguments.overwrite,
    )
    print(
        "INT8 测试集评估完成："
        f"FP32 accuracy={result.float_metrics.accuracy:.4%}，"
        f"INT8 accuracy={result.int8_metrics.accuracy:.4%}，"
        f"预测一致率={result.prediction_agreement:.4%}"
    )
    print(f"评估产物：{result.metrics_path.parent.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "INT8_EVALUATION_FORMAT_VERSION",
    "Int8TFLiteEvaluationResult",
    "evaluate_int8_tflite",
    "main",
]
