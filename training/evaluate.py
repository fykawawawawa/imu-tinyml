"""在 UCI HAR 官方测试集上评估最佳 TinyMLP 检查点。

评估前会校验检查点中的模型结构、39 维特征顺序、六类活动标签和标准化参数，防止把
不兼容的数据或模型组合在一起。测试 DataLoader 不打乱、不丢弃样本，推理全程关闭梯度。

默认在 ``artifacts/evaluation`` 中生成以下文件：

* ``metrics.json``：总体指标、逐类别指标、模型信息和最佳验证性能；
* ``classification_report.csv``：六类活动及宏平均、加权平均指标；
* ``confusion_matrix.csv``：行是真实类别、列是预测类别的原始计数；
* ``confusion_matrix.png``：同时标注计数和按真实类别归一化百分比的热力图；
* ``evaluation.log``：数据契约、模型信息和最终测试性能日志。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES, TinyMLP, count_trainable_parameters
from training.data import ACTIVITY_NAMES
from training.preprocess import (
    SPLIT_FILENAMES,
    STANDARDIZATION_FILENAME,
    ProcessedFeatureDataset,
    load_processed_split,
    load_standardization_parameters,
)

# 评估可能在没有桌面显示服务的 CI 或服务器中运行，必须使用无界面的绘图后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DeviceName = Literal["auto", "cpu", "cuda"]
EVALUATION_FORMAT_VERSION = 1
DEFAULT_METRICS_FILENAME = "metrics.json"
DEFAULT_REPORT_FILENAME = "classification_report.csv"
DEFAULT_CONFUSION_CSV_FILENAME = "confusion_matrix.csv"
DEFAULT_CONFUSION_PNG_FILENAME = "confusion_matrix.png"
DEFAULT_LOG_FILENAME = "evaluation.log"


@dataclass(frozen=True)
class EvaluationConfig:
    """测试集推理配置。"""

    batch_size: int = 256
    num_workers: int = 0
    device: DeviceName = "auto"

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size 必须是正整数")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers 必须是非负整数")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 只能是 'auto'、'cpu' 或 'cuda'")


@dataclass(frozen=True)
class ClassMetrics:
    """单个活动类别的测试指标。"""

    label: int
    activity: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """测试集总体指标与逐类别指标。"""

    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    sample_count: int
    class_metrics: tuple[ClassMetrics, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """一次正式评估的结果及全部产物路径。"""

    metrics: EvaluationMetrics
    confusion_matrix: Tensor
    metrics_path: Path
    report_path: Path
    confusion_csv_path: Path
    confusion_png_path: Path
    log_path: Path


def _resolve_device(requested: DeviceName) -> torch.device:
    """解析评估设备，并拒绝不可用的显式 CUDA 请求。"""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA 评估，但当前环境没有可用 CUDA 设备")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_worker(worker_id: int) -> None:
    """固定 DataLoader 工作进程的 NumPy 随机状态。"""
    del worker_id
    np.random.seed(torch.initial_seed() % (2**32))


def _create_test_loader(
    dataset: ProcessedFeatureDataset,
    *,
    config: EvaluationConfig,
    pin_memory: bool,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """创建不打乱、不丢弃样本的测试 DataLoader。"""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        persistent_workers=config.num_workers > 0,
    )


def _require_mapping(payload: object, *, name: str) -> dict[str, object]:
    """确认检查点字段为字符串键映射。"""
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"检查点字段 {name} 必须是字符串键映射")
    return payload


def load_best_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[TinyMLP, dict[str, object]]:
    """安全加载并严格校验最佳模型检查点。

    使用 ``weights_only=True`` 限制反序列化范围。模型名称、输入维度、隐藏层、类别数、
    特征顺序和活动顺序必须与当前源码完全一致，任何一项不符都会在测试推理前失败。
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到最佳模型检查点：{path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"无法加载检查点 {path}：{error}") from error
    checkpoint = _require_mapping(checkpoint, name="root")

    required_keys = {
        "epoch",
        "model_name",
        "model_config",
        "model_state_dict",
        "performance",
        "feature_names",
        "activity_names",
        "standardization",
    }
    missing = sorted(required_keys.difference(checkpoint))
    if missing:
        raise ValueError(f"检查点缺少字段：{', '.join(missing)}")
    if checkpoint["model_name"] != "TinyMLP":
        raise ValueError(f"检查点模型必须是 TinyMLP，实际为 {checkpoint['model_name']!r}")

    model_config = _require_mapping(checkpoint["model_config"], name="model_config")
    expected_model_config = {
        "input_dim": FEATURE_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "num_classes": NUM_CLASSES,
    }
    for name, expected in expected_model_config.items():
        actual = model_config.get(name)
        if name == "hidden_dims" and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected:
            raise ValueError(
                f"检查点模型配置 {name} 应为 {expected!r}，实际为 {actual!r}"
            )

    stored_feature_names = tuple(str(name) for name in checkpoint["feature_names"])
    if stored_feature_names != FEATURE_NAMES:
        raise ValueError("检查点的特征名称或顺序与当前 FEATURE_NAMES 不一致")
    stored_activity_names = tuple(str(name) for name in checkpoint["activity_names"])
    if stored_activity_names != ACTIVITY_NAMES:
        raise ValueError("检查点的活动标签或顺序与当前 ACTIVITY_NAMES 不一致")

    state_dict = _require_mapping(checkpoint["model_state_dict"], name="model_state_dict")
    model = TinyMLP()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(f"模型权重与当前 TinyMLP 结构不兼容：{error}") from error
    model.to(device)
    model.eval()
    return model, checkpoint


def _validate_standardization_contract(
    checkpoint: dict[str, object],
    processed_dir: Path,
) -> None:
    """确认测试特征使用的标准化参数与训练检查点完全一致。"""
    saved_parameters = load_standardization_parameters(
        processed_dir / STANDARDIZATION_FILENAME
    )
    checkpoint_parameters = _require_mapping(
        checkpoint["standardization"], name="standardization"
    )
    if "mean" not in checkpoint_parameters or "scale" not in checkpoint_parameters:
        raise ValueError("检查点标准化参数缺少 mean 或 scale")
    checkpoint_mean = torch.as_tensor(checkpoint_parameters["mean"]).cpu().numpy()
    checkpoint_scale = torch.as_tensor(checkpoint_parameters["scale"]).cpu().numpy()
    if not np.array_equal(checkpoint_mean, saved_parameters.mean):
        raise ValueError("检查点 mean 与测试数据使用的标准化 mean 不一致")
    if not np.array_equal(checkpoint_scale, saved_parameters.scale):
        raise ValueError("检查点 scale 与测试数据使用的标准化 scale 不一致")


def metrics_from_confusion_matrix(
    confusion: Tensor,
    *,
    loss: float,
) -> EvaluationMetrics:
    """从六类混淆矩阵计算总体、宏平均、加权平均和逐类别指标。"""
    if confusion.ndim != 2 or confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            f"confusion 的形状必须为 ({NUM_CLASSES}, {NUM_CLASSES})，"
            f"实际为 {tuple(confusion.shape)}"
        )
    if torch.any(confusion < 0):
        raise ValueError("confusion 不能包含负数")
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError("loss 必须是有限非负数")

    values = confusion.to(dtype=torch.float64, device="cpu")
    support = values.sum(dim=1)
    predicted = values.sum(dim=0)
    true_positive = torch.diag(values)
    sample_count = int(values.sum().item())
    if sample_count <= 0:
        raise ValueError("confusion 至少需要包含一个样本")

    precision = torch.where(
        predicted > 0.0,
        true_positive / predicted,
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        support > 0.0,
        true_positive / support,
        torch.zeros_like(true_positive),
    )
    f1 = torch.where(
        precision + recall > 0.0,
        2.0 * precision * recall / (precision + recall),
        torch.zeros_like(true_positive),
    )
    weights = support / sample_count

    per_class = tuple(
        ClassMetrics(
            label=index,
            activity=ACTIVITY_NAMES[index],
            precision=float(precision[index].item()),
            recall=float(recall[index].item()),
            f1=float(f1[index].item()),
            support=int(support[index].item()),
        )
        for index in range(NUM_CLASSES)
    )
    return EvaluationMetrics(
        loss=float(loss),
        accuracy=float(true_positive.sum().item() / sample_count),
        macro_precision=float(precision.mean().item()),
        macro_recall=float(recall.mean().item()),
        macro_f1=float(f1.mean().item()),
        weighted_precision=float(torch.sum(precision * weights).item()),
        weighted_recall=float(torch.sum(recall * weights).item()),
        weighted_f1=float(torch.sum(f1 * weights).item()),
        sample_count=sample_count,
        class_metrics=per_class,
    )


def _run_inference(
    model: TinyMLP,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[EvaluationMetrics, Tensor]:
    """在完整测试集上推理并累计交叉熵与混淆矩阵。"""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    total_loss = 0.0
    model.eval()

    with torch.inference_mode():
        for features, labels, _subjects in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            if not torch.all(torch.isfinite(logits)):
                raise RuntimeError("模型输出包含 NaN 或 Inf")
            total_loss += float(criterion(logits, labels).item())

            predictions = torch.argmax(logits, dim=1)
            flat_indices = labels * NUM_CLASSES + predictions
            confusion += torch.bincount(
                flat_indices.to(device="cpu"), minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    sample_count = int(confusion.sum().item())
    if sample_count == 0:
        raise RuntimeError("测试 DataLoader 没有产生任何样本")
    return metrics_from_confusion_matrix(
        confusion, loss=total_loss / sample_count
    ), confusion


def _configure_logger(log_path: Path) -> logging.Logger:
    """创建同时输出到控制台和 UTF-8 文件的评估日志器。"""
    logger = logging.getLogger(f"imu_tinyml.evaluation.{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    """刷新并关闭日志 handler，避免 Windows 持续占用输出文件。"""
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()


def _atomic_save_json(path: Path, payload: dict[str, object]) -> None:
    """原子保存 UTF-8 JSON 文件。"""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_csv(path: Path, rows: Sequence[Sequence[object]]) -> None:
    """原子保存带 UTF-8 BOM 的 CSV，确保 Excel 正确识别活动名称。"""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_classification_report(path: Path, metrics: EvaluationMetrics) -> None:
    """保存逐类别、宏平均、加权平均及准确率报告。"""
    rows: list[Sequence[object]] = [
        ("label", "activity", "precision", "recall", "f1", "support")
    ]
    rows.extend(
        (
            item.label,
            item.activity,
            f"{item.precision:.8f}",
            f"{item.recall:.8f}",
            f"{item.f1:.8f}",
            item.support,
        )
        for item in metrics.class_metrics
    )
    rows.extend(
        (
            (
                "macro_avg",
                "MACRO_AVERAGE",
                f"{metrics.macro_precision:.8f}",
                f"{metrics.macro_recall:.8f}",
                f"{metrics.macro_f1:.8f}",
                metrics.sample_count,
            ),
            (
                "weighted_avg",
                "WEIGHTED_AVERAGE",
                f"{metrics.weighted_precision:.8f}",
                f"{metrics.weighted_recall:.8f}",
                f"{metrics.weighted_f1:.8f}",
                metrics.sample_count,
            ),
            (
                "accuracy",
                "ACCURACY",
                "",
                "",
                f"{metrics.accuracy:.8f}",
                metrics.sample_count,
            ),
        )
    )
    _atomic_save_csv(path, rows)


def _save_confusion_csv(path: Path, confusion: Tensor) -> None:
    """保存带真实/预测方向说明的原始混淆矩阵。"""
    values = confusion.to(dtype=torch.int64, device="cpu").numpy()
    rows: list[Sequence[object]] = [
        ("true_label\\predicted_label", *ACTIVITY_NAMES)
    ]
    rows.extend(
        (ACTIVITY_NAMES[row_index], *values[row_index].tolist())
        for row_index in range(NUM_CLASSES)
    )
    _atomic_save_csv(path, rows)


def _save_confusion_figure(
    path: Path,
    confusion: Tensor,
    *,
    accuracy: float,
) -> None:
    """绘制计数与按行百分比并存的混淆矩阵热力图。"""
    counts = confusion.to(dtype=torch.int64, device="cpu").numpy()
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(10.5, 8.2))
    image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Recall-normalized proportion")
    axis.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=ACTIVITY_NAMES,
        yticklabels=ACTIVITY_NAMES,
        xlabel="Predicted activity",
        ylabel="True activity",
        title=(
            "UCI HAR Official Test Set Confusion Matrix\n"
            f"TinyMLP | {FEATURE_DIM} features | Accuracy {accuracy:.2%}"
        ),
    )
    plt.setp(axis.get_xticklabels(), rotation=32, ha="right", rotation_mode="anchor")

    # 深色格使用白字、浅色格使用深色字，兼顾打印和屏幕阅读。
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            color = "white" if normalized[row, column] >= 0.5 else "#17212b"
            axis.text(
                column,
                row,
                f"{counts[row, column]}\n{normalized[row, column]:.1%}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    axis.set_ylim(NUM_CLASSES - 0.5, -0.5)
    figure.tight_layout()
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        figure.savefig(temporary_path, format="png", dpi=180, bbox_inches="tight")
        os.replace(temporary_path, path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


def evaluate_best_model(
    checkpoint_path: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path,
    *,
    config: EvaluationConfig | None = None,
    overwrite: bool = False,
) -> EvaluationResult:
    """在官方测试集上评估最佳模型并保存全部报告产物。"""
    evaluation_config = EvaluationConfig() if config is None else config
    if not isinstance(evaluation_config, EvaluationConfig):
        raise TypeError("config 必须是 EvaluationConfig 或 None")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值")

    source = Path(processed_dir).expanduser()
    checkpoint_file = Path(checkpoint_path).expanduser()
    destination = Path(output_dir).expanduser()
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

    test_dataset = load_processed_split(source / SPLIT_FILENAMES["test"])
    device = _resolve_device(evaluation_config.device)
    model, checkpoint = load_best_model(checkpoint_file, device=device)
    _validate_standardization_contract(checkpoint, source)
    test_loader = _create_test_loader(
        test_dataset,
        config=evaluation_config,
        pin_memory=device.type == "cuda",
    )

    destination.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(log_path)
    try:
        model_config = _require_mapping(checkpoint["model_config"], name="model_config")
        logger.info(
            "官方测试集评估 | 特征维度=%d | 模型=TinyMLP | 结构=%d -> %d -> %d -> %d "
            "| 可训练参数=%d",
            FEATURE_DIM,
            FEATURE_DIM,
            HIDDEN_DIMS[0],
            HIDDEN_DIMS[1],
            NUM_CLASSES,
            model_config.get("parameter_count", count_trainable_parameters(model)),
        )
        logger.info(
            "检查点=%s | 最佳训练轮次=%s | 测试样本=%d | 测试受试者=%d | 设备=%s",
            checkpoint_file.resolve(),
            checkpoint["epoch"],
            len(test_dataset),
            len(set(test_dataset.subjects.tolist())),
            device,
        )
        metrics, confusion = _run_inference(model, test_loader, device)
        logger.info(
            "测试总体性能 | loss=%.6f | accuracy=%.2f%% | macro_precision=%.4f | "
            "macro_recall=%.4f | macro_f1=%.4f | weighted_f1=%.4f",
            metrics.loss,
            metrics.accuracy * 100.0,
            metrics.macro_precision,
            metrics.macro_recall,
            metrics.macro_f1,
            metrics.weighted_f1,
        )
        for item in metrics.class_metrics:
            logger.info(
                "类别性能 | label=%d | activity=%s | precision=%.4f | recall=%.4f | "
                "f1=%.4f | support=%d",
                item.label,
                item.activity,
                item.precision,
                item.recall,
                item.f1,
                item.support,
            )

        validation_performance = _require_mapping(
            checkpoint["performance"], name="performance"
        )
        metrics_payload: dict[str, object] = {
            "format_version": EVALUATION_FORMAT_VERSION,
            "task": "UCI HAR 六类运动状态分类",
            "evaluation_split": "official_test",
            "checkpoint": str(checkpoint_file.resolve()),
            "best_epoch": int(checkpoint["epoch"]),
            "model": {
                "name": "TinyMLP",
                "feature_dim": FEATURE_DIM,
                "structure": [FEATURE_DIM, *HIDDEN_DIMS, NUM_CLASSES],
                "parameter_count": int(
                    model_config.get("parameter_count", count_trainable_parameters(model))
                ),
            },
            "feature_names": list(FEATURE_NAMES),
            "activity_names": list(ACTIVITY_NAMES),
            "evaluation_config": asdict(evaluation_config),
            "best_validation_performance": validation_performance,
            "test": {
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "macro_precision": metrics.macro_precision,
                "macro_recall": metrics.macro_recall,
                "macro_f1": metrics.macro_f1,
                "weighted_precision": metrics.weighted_precision,
                "weighted_recall": metrics.weighted_recall,
                "weighted_f1": metrics.weighted_f1,
                "sample_count": metrics.sample_count,
                "subject_count": len(set(test_dataset.subjects.tolist())),
                "per_class": [asdict(item) for item in metrics.class_metrics],
            },
            "confusion_matrix_orientation": {
                "rows": "true_activity",
                "columns": "predicted_activity",
            },
        }
        _atomic_save_json(metrics_path, metrics_payload)
        _save_classification_report(report_path, metrics)
        _save_confusion_csv(confusion_csv_path, confusion)
        _save_confusion_figure(
            confusion_png_path, confusion, accuracy=metrics.accuracy
        )
        logger.info("评估产物已保存到 %s", destination.resolve())
    finally:
        _close_logger(logger)

    return EvaluationResult(
        metrics=metrics,
        confusion_matrix=confusion,
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
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/best_model.pt"),
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有评估输出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    arguments = _build_argument_parser().parse_args(argv)
    result = evaluate_best_model(
        arguments.checkpoint,
        arguments.processed_dir,
        arguments.output_dir,
        config=EvaluationConfig(
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            device=arguments.device,
        ),
        overwrite=arguments.overwrite,
    )
    print(
        "测试集评估完成："
        f"accuracy={result.metrics.accuracy:.2%}, "
        f"macro_f1={result.metrics.macro_f1:.4f}, "
        f"loss={result.metrics.loss:.6f}"
    )
    print(f"评估产物：{result.metrics_path.parent.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFUSION_CSV_FILENAME",
    "DEFAULT_CONFUSION_PNG_FILENAME",
    "DEFAULT_LOG_FILENAME",
    "DEFAULT_METRICS_FILENAME",
    "DEFAULT_REPORT_FILENAME",
    "ClassMetrics",
    "EvaluationConfig",
    "EvaluationMetrics",
    "EvaluationResult",
    "evaluate_best_model",
    "load_best_model",
    "main",
    "metrics_from_confusion_matrix",
]
