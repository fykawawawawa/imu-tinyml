"""训练并验证用于六类运动状态识别的轻量 MLP。

本模块读取 :mod:`training.preprocess` 生成的 39 维标准化特征，训练现有的
``TinyMLP(39 -> 32 -> 16 -> 6)``。每轮日志同时记录损失、准确率以及宏平均
precision/recall/F1；验证损失连续若干轮没有改善时提前停止，并保存验证损失最低的模型。

最佳检查点不仅包含模型权重，还包含标准化参数、固定特征顺序、活动标签映射、模型结构、
训练配置和最佳轮性能，供评估、量化与部署代码校验数据契约。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES, TinyMLP, count_trainable_parameters
from training.data import ACTIVITY_NAMES
from training.preprocess import (
    METADATA_FILENAME,
    SPLIT_FILENAMES,
    STANDARDIZATION_FILENAME,
    ProcessedFeatureDataset,
    StandardizationParameters,
    load_processed_split,
    load_standardization_parameters,
)

DeviceName = Literal["auto", "cpu", "cuda"]
CHECKPOINT_FORMAT_VERSION = 1
DEFAULT_CHECKPOINT_FILENAME = "best_model.pt"
DEFAULT_HISTORY_FILENAME = "history.json"
DEFAULT_LOG_FILENAME = "training.log"
DEFAULT_CONFIG_PATH = Path("configs/train.yaml")


@dataclass(frozen=True)
class TrainingConfig:
    """训练超参数及可复现性配置。"""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    min_delta: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    device: DeviceName = "auto"

    def __post_init__(self) -> None:
        positive_integers = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers 必须是非负整数")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed 必须是非负整数")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须是有限正数")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay 必须是有限非负数")
        if not np.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("min_delta 必须是有限非负数")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 只能是 'auto'、'cpu' 或 'cuda'")


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """从 YAML 文件加载训练参数，并交由 :class:`TrainingConfig` 校验。

    YAML 文件只负责保存易于修改和复现的实验参数；字段类型、默认值和合法性规则仍集中
    定义在 ``TrainingConfig`` 中。配置可以省略希望使用类默认值的字段，但不允许出现
    未知字段，以免参数名拼写错误后被静默忽略。

    参数：
        config_path: YAML 配置文件路径。相对路径以运行命令时的工作目录为基准。

    返回：
        完成类型与取值校验的 ``TrainingConfig``。

    异常：
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 无法解析、根节点不是映射、包含未知字段或参数值不合法。
    """
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到训练配置文件：{path}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"无法解析训练配置文件 {path}：{error}") from error
    except OSError as error:
        raise ValueError(f"无法读取训练配置文件 {path}：{error}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"训练配置文件 {path} 的根节点必须是键值映射")

    allowed_fields = {field.name for field in fields(TrainingConfig)}
    unknown_fields = sorted(set(payload).difference(allowed_fields))
    if unknown_fields:
        raise ValueError(
            f"训练配置文件 {path} 包含未知字段：{', '.join(unknown_fields)}"
        )

    try:
        return TrainingConfig(**payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"训练配置文件 {path} 的参数无效：{error}") from error


@dataclass(frozen=True)
class ClassificationMetrics:
    """由六类混淆矩阵计算得到的分类性能。"""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


@dataclass(frozen=True)
class EpochRecord:
    """单轮训练与验证结果。"""

    epoch: int
    train_loss: float
    train_accuracy: float
    train_macro_f1: float
    val_loss: float
    val_accuracy: float
    val_macro_precision: float
    val_macro_recall: float
    val_macro_f1: float
    improved: bool


@dataclass(frozen=True)
class TrainingResult:
    """完整训练运行的输出位置和最佳性能摘要。"""

    checkpoint_path: Path
    history_path: Path
    log_path: Path
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    best_val_macro_f1: float
    epochs_completed: int
    stopped_early: bool


def _set_reproducible_seed(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 以及 CUDA 的随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 固定 cuDNN 算法选择；当前 MLP 通常不调用 cuDNN，但保留完整约定。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """为 DataLoader 子进程设置彼此不同但可复现的随机种子。"""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resolve_device(requested: DeviceName) -> torch.device:
    """解析训练设备，并在显式请求不可用 CUDA 时给出清晰错误。"""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA 训练，但当前环境没有可用 CUDA 设备")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _create_data_loader(
    dataset: ProcessedFeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """为已处理特征创建可复现的 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
        worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )


def classification_metrics_from_confusion(confusion: Tensor) -> ClassificationMetrics:
    """从方形混淆矩阵计算 accuracy 和六类宏平均指标。

    行表示真实标签，列表示预测标签。某类别没有预测样本或真实样本时，其对应 precision
    或 recall 按 0 处理；宏平均始终覆盖全部六类，而不是只平均当前批次出现的类别。
    """
    if confusion.ndim != 2 or confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            f"confusion 的形状必须为 ({NUM_CLASSES}, {NUM_CLASSES})，"
            f"实际为 {tuple(confusion.shape)}"
        )
    if torch.any(confusion < 0):
        raise ValueError("confusion 不能包含负数")

    values = confusion.to(dtype=torch.float64, device="cpu")
    sample_count = float(values.sum().item())
    if sample_count <= 0.0:
        raise ValueError("confusion 至少需要包含一个样本")

    true_positive = torch.diag(values)
    predicted_count = values.sum(dim=0)
    actual_count = values.sum(dim=1)
    precision = torch.where(
        predicted_count > 0.0,
        true_positive / predicted_count,
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        actual_count > 0.0,
        true_positive / actual_count,
        torch.zeros_like(true_positive),
    )
    f1 = torch.where(
        precision + recall > 0.0,
        2.0 * precision * recall / (precision + recall),
        torch.zeros_like(true_positive),
    )
    return ClassificationMetrics(
        accuracy=float(true_positive.sum().item() / sample_count),
        macro_precision=float(precision.mean().item()),
        macro_recall=float(recall.mean().item()),
        macro_f1=float(f1.mean().item()),
    )


def _run_epoch(
    model: TinyMLP,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    *,
    optimizer: Optimizer | None,
) -> tuple[float, ClassificationMetrics]:
    """运行一轮训练或只读验证，并累计完整分区的分类指标。"""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    sample_count = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    # 验证阶段关闭 autograd，减少内存和算力；训练阶段保持正常梯度记录。
    with torch.set_grad_enabled(is_training):
        for features, labels, _subjects in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size

            predictions = torch.argmax(logits.detach(), dim=1)
            flat_indices = labels.detach() * NUM_CLASSES + predictions
            batch_confusion = torch.bincount(
                flat_indices.to(device="cpu"), minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            confusion += batch_confusion

    if sample_count == 0:
        raise RuntimeError("DataLoader 没有产生任何样本")
    return total_loss / sample_count, classification_metrics_from_confusion(confusion)


def _configure_logger(log_path: Path) -> logging.Logger:
    """创建同时写入控制台和 UTF-8 文件的独立日志器。"""
    logger = logging.getLogger(f"imu_tinyml.training.{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 同一 Python 进程重复训练时，先关闭旧 handler，避免日志重复及 Windows 文件占用。
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
    """刷新并关闭本次训练创建的日志文件。"""
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()


def _atomic_save_json(path: Path, payload: dict[str, object]) -> None:
    """原子写入 JSON，避免训练中断时损坏历史记录。"""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    """原子保存 PyTorch 检查点。"""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_preprocessing_metadata(processed_dir: Path) -> dict[str, object] | None:
    """读取可选预处理元数据；旧产物没有该文件时仍允许训练。"""
    metadata_path = processed_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取预处理元数据 {metadata_path}：{error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"预处理元数据 {metadata_path} 的根节点必须是对象")
    stored_names = payload.get("feature_names")
    if stored_names is not None and stored_names != list(FEATURE_NAMES):
        raise ValueError("预处理元数据中的特征名称或顺序与当前 FEATURE_NAMES 不一致")
    return payload


def _checkpoint_payload(
    *,
    model: TinyMLP,
    optimizer: Optimizer,
    standardization: StandardizationParameters,
    config: TrainingConfig,
    record: EpochRecord,
    preprocessing_metadata: dict[str, object] | None,
) -> dict[str, object]:
    """构造自描述、可跨 CPU/GPU 加载的最佳模型检查点。"""
    cpu_state_dict = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": record.epoch,
        "model_name": "TinyMLP",
        "model_config": {
            "input_dim": FEATURE_DIM,
            "hidden_dims": HIDDEN_DIMS,
            "num_classes": NUM_CLASSES,
            "parameter_count": count_trainable_parameters(model),
        },
        "model_state_dict": cpu_state_dict,
        "optimizer_name": "Adam",
        "optimizer_state_dict": optimizer.state_dict(),
        "training_config": asdict(config),
        "performance": asdict(record),
        "feature_names": FEATURE_NAMES,
        "activity_names": ACTIVITY_NAMES,
        "label_mapping": {index: name for index, name in enumerate(ACTIVITY_NAMES)},
        "standardization": {
            "mean": torch.from_numpy(standardization.mean.copy()),
            "scale": torch.from_numpy(standardization.scale.copy()),
            "minimum_scale": standardization.minimum_scale,
        },
        "preprocessing_metadata": preprocessing_metadata,
    }


def train_model(
    processed_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TrainingConfig | None = None,
    overwrite: bool = False,
) -> TrainingResult:
    """训练 TinyMLP、执行验证和早停，并保存最佳检查点与日志。

    参数：
        processed_dir: :func:`training.preprocess.preprocess_uci_har` 的输出目录。
        output_dir: 检查点、训练历史和文本日志的保存目录。
        config: 训练超参数；省略时使用 :class:`TrainingConfig` 默认值。
        overwrite: 是否允许覆盖同名输出。默认拒绝覆盖，避免误删已有实验结果。
    """
    training_config = TrainingConfig() if config is None else config
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("config 必须是 TrainingConfig 或 None")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值")

    source = Path(processed_dir).expanduser()
    destination = Path(output_dir).expanduser()
    checkpoint_path = destination / DEFAULT_CHECKPOINT_FILENAME
    history_path = destination / DEFAULT_HISTORY_FILENAME
    log_path = destination / DEFAULT_LOG_FILENAME
    existing = [path for path in (checkpoint_path, history_path, log_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"训练输出已存在；如需覆盖请设置 overwrite=True：{joined}")

    train_dataset = load_processed_split(source / SPLIT_FILENAMES["train"])
    validation_dataset = load_processed_split(source / SPLIT_FILENAMES["val"])
    standardization = load_standardization_parameters(source / STANDARDIZATION_FILENAME)
    preprocessing_metadata = _read_preprocessing_metadata(source)

    _set_reproducible_seed(training_config.seed)
    device = _resolve_device(training_config.device)
    model = TinyMLP().to(device)
    parameter_count = count_trainable_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    use_pin_memory = device.type == "cuda"
    train_loader = _create_data_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=training_config.seed,
        num_workers=training_config.num_workers,
        pin_memory=use_pin_memory,
    )
    validation_loader = _create_data_loader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
        num_workers=training_config.num_workers,
        pin_memory=use_pin_memory,
    )

    destination.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(log_path)
    history: list[EpochRecord] = []
    best_record: EpochRecord | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    try:
        # 这两行是实验日志的数据契约：明确记录用户要求的特征维度、轻量模型及分类任务。
        logger.info(
            "运动状态分类配置 | 特征维度=%d | 模型=TinyMLP | 结构=%d -> %d -> %d -> %d "
            "| 可训练参数=%d",
            FEATURE_DIM,
            FEATURE_DIM,
            HIDDEN_DIMS[0],
            HIDDEN_DIMS[1],
            NUM_CLASSES,
            parameter_count,
        )
        logger.info(
            "运动类别=%s | 训练样本=%d | 验证样本=%d | 设备=%s",
            ", ".join(f"{index}:{name}" for index, name in enumerate(ACTIVITY_NAMES)),
            len(train_dataset),
            len(validation_dataset),
            device,
        )
        logger.info(
            "训练参数 | epochs=%d | batch_size=%d | learning_rate=%.6g | "
            "weight_decay=%.6g | patience=%d | min_delta=%.6g | seed=%d",
            training_config.epochs,
            training_config.batch_size,
            training_config.learning_rate,
            training_config.weight_decay,
            training_config.patience,
            training_config.min_delta,
            training_config.seed,
        )

        for epoch in range(1, training_config.epochs + 1):
            train_loss, train_metrics = _run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer
            )
            val_loss, val_metrics = _run_epoch(
                model, validation_loader, criterion, device, optimizer=None
            )
            #  模型早停
            improved = val_loss < best_val_loss - training_config.min_delta
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_metrics.accuracy,
                train_macro_f1=train_metrics.macro_f1,
                val_loss=val_loss,
                val_accuracy=val_metrics.accuracy,
                val_macro_precision=val_metrics.macro_precision,
                val_macro_recall=val_metrics.macro_recall,
                val_macro_f1=val_metrics.macro_f1,
                improved=improved,
            )
            history.append(record)

            marker = " | 已保存最佳模型" if improved else ""
            logger.info(
                "Epoch %03d/%03d | train_loss=%.6f | train_accuracy=%.2f%% | "
                "train_macro_f1=%.4f | val_loss=%.6f | val_accuracy=%.2f%% | "
                "val_macro_precision=%.4f | val_macro_recall=%.4f | "
                "val_macro_f1=%.4f%s",
                epoch,
                training_config.epochs,
                train_loss,
                train_metrics.accuracy * 100.0,
                train_metrics.macro_f1,
                val_loss,
                val_metrics.accuracy * 100.0,
                val_metrics.macro_precision,
                val_metrics.macro_recall,
                val_metrics.macro_f1,
                marker,
            )

            if improved:
                best_val_loss = val_loss
                best_record = record
                epochs_without_improvement = 0
                checkpoint = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    standardization=standardization,
                    config=training_config,
                    record=record,
                    preprocessing_metadata=preprocessing_metadata,
                )
                _atomic_save_checkpoint(checkpoint_path, checkpoint)
            else:
                epochs_without_improvement += 1

            history_payload: dict[str, object] = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "task": "UCI HAR 六类运动状态分类",
                "feature_dim": FEATURE_DIM,
                "feature_names": list(FEATURE_NAMES),
                "model": {
                    "name": "TinyMLP",
                    "structure": [FEATURE_DIM, *HIDDEN_DIMS, NUM_CLASSES],
                    "parameter_count": parameter_count,
                },
                "activity_names": list(ACTIVITY_NAMES),
                "training_config": asdict(training_config),
                "epochs": [asdict(item) for item in history],
                "best_epoch": None if best_record is None else best_record.epoch,
                "stopped_early": False,
            }
            _atomic_save_json(history_path, history_payload)

            if epochs_without_improvement >= training_config.patience:
                stopped_early = True
                logger.info(
                    "早停触发 | 连续 %d 轮验证损失未改善 | 最佳轮次=%d",
                    training_config.patience,
                    best_record.epoch if best_record is not None else -1,
                )
                break

        if best_record is None:
            raise RuntimeError("训练未产生有效的最佳检查点")

        # 训练结束后补写最终早停状态；各轮记录已在训练过程中持续落盘。
        final_history: dict[str, object] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "task": "UCI HAR 六类运动状态分类",
            "feature_dim": FEATURE_DIM,
            "feature_names": list(FEATURE_NAMES),
            "model": {
                "name": "TinyMLP",
                "structure": [FEATURE_DIM, *HIDDEN_DIMS, NUM_CLASSES],
                "parameter_count": parameter_count,
            },
            "activity_names": list(ACTIVITY_NAMES),
            "training_config": asdict(training_config),
            "epochs": [asdict(item) for item in history],
            "best_epoch": best_record.epoch,
            "stopped_early": stopped_early,
        }
        _atomic_save_json(history_path, final_history)
        logger.info(
            "训练完成 | 已运行轮次=%d | 最佳轮次=%d | 最佳验证损失=%.6f | "
            "最佳验证准确率=%.2f%% | 最佳验证macro_f1=%.4f | 检查点=%s",
            len(history),
            best_record.epoch,
            best_record.val_loss,
            best_record.val_accuracy * 100.0,
            best_record.val_macro_f1,
            checkpoint_path.resolve(),
        )
    finally:
        _close_logger(logger)

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        log_path=log_path,
        best_epoch=best_record.epoch,
        best_val_loss=best_record.val_loss,
        best_val_accuracy=best_record.val_accuracy,
        best_val_macro_f1=best_record.val_macro_f1,
        epochs_completed=len(history),
        stopped_early=stopped_early,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"训练参数 YAML 文件（默认：{DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有训练输出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    arguments = _build_argument_parser().parse_args(argv)
    config = load_training_config(arguments.config)
    result = train_model(
        arguments.processed_dir,
        arguments.output_dir,
        config=config,
        overwrite=arguments.overwrite,
    )
    print(
        "训练完成："
        f"best_epoch={result.best_epoch}, "
        f"val_accuracy={result.best_val_accuracy:.2%}, "
        f"val_macro_f1={result.best_val_macro_f1:.4f}"
    )
    print(f"最佳检查点：{result.checkpoint_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CHECKPOINT_FILENAME",
    "DEFAULT_HISTORY_FILENAME",
    "DEFAULT_LOG_FILENAME",
    "ClassificationMetrics",
    "EpochRecord",
    "TrainingConfig",
    "TrainingResult",
    "classification_metrics_from_confusion",
    "load_training_config",
    "main",
    "train_model",
]
