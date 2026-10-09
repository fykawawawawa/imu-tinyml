"""将 PyTorch TinyMLP 检查点导出为等价的 Keras Dense/ReLU 模型。

导出过程严格复用当前项目的模型与检查点契约，将 PyTorch ``Linear`` 的
``[out_features, in_features]`` 权重转置为 Keras ``Dense`` 的
``[in_features, out_features]`` kernel。模型输出保持为六类 logits，不添加 Softmax。

保存前后都会在处理后的测试特征上与 PyTorch 基线比较；只有 logits 满足数值容差且
所有预测类别完全一致时，才会提交 ``.keras`` 文件和转换报告。

转换分为四个阶段：

1. 通过训练阶段已有的检查点加载函数校验模型结构、特征顺序和活动标签；
2. 创建算子顺序完全对应的 Keras 网络，并逐层迁移权重；
3. 使用官方测试集比较内存中 PyTorch/Keras 模型的 logits；
4. 将 Keras 模型保存、重新加载并再次对齐，排除序列化造成的差异。

这里生成的 Keras 模型用于 TFLite 转换，不进行重新训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES
from training.data import ACTIVITY_NAMES
from training.evaluate import load_best_model
from training.preprocess import SPLIT_FILENAMES, load_processed_split

if TYPE_CHECKING:
    # Model 只用于类型标注。运行时延迟导入 TensorFlow/Keras，避免普通训练和评估命令
    # 因未安装 convert 可选依赖而在模块导入阶段失败。
    from keras import Model

# 独立版本号用于识别报告格式。
KERAS_EXPORT_FORMAT_VERSION = 1
# 模型文件与报告都放在 ignored 的生成物目录中，避免二进制模型误提交到 Git。
DEFAULT_KERAS_MODEL_PATH = Path("models/exported/best_model.keras")
DEFAULT_CONVERSION_REPORT_PATH = Path("models/exported/best_model_keras_report.json")


@dataclass(frozen=True)
class KerasExportResult:
    """Keras 导出文件及 PyTorch/Keras 对齐结果。

    该对象只保存调用方最常用的摘要；完整的层映射、依赖版本和两次验证结果会写入
    ``report_path`` 指向的 JSON 文件。
    """

    model_path: Path
    report_path: Path
    sample_count: int
    max_absolute_error: float
    mean_absolute_error: float
    prediction_agreement: float
    model_sha256: str


def _import_tensorflow():
    """延迟导入 TensorFlow，使不使用转换功能的训练环境无需安装它。

    TensorFlow 是项目 ``convert`` 可选依赖，而 PyTorch 训练只需要 ``train`` 依赖。
    将导入放在函数内部可以保持两个运行环境解耦，同时给缺少依赖的用户更明确的错误。
    """
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Keras 导出需要 TensorFlow；请安装项目的 convert 或 all 可选依赖"
        ) from error
    return tf


def build_keras_tiny_mlp() -> Model:
    """创建与 :class:`models.mlp.TinyMLP` 结构等价的 Keras 模型。

    ReLU 使用独立层而不是 ``Dense(activation="relu")``，这样层级结构与 PyTorch 的
    ``Linear/ReLU`` 顺序一一对应，也便于转换报告和测试明确检查每个算子。最后一层不加
    Softmax，因为原模型输出的是 logits，部署时只需 ``argmax``。
    """
    tf = _import_tensorflow()
    return tf.keras.Sequential(
        [
            # None 表示动态批量维；单个样本和整批样本都共享固定的 39 维特征契约。
            tf.keras.layers.Input(shape=(FEATURE_DIM,), dtype="float32", name="features"),
            tf.keras.layers.Dense(HIDDEN_DIMS[0], name="dense_1"),
            tf.keras.layers.ReLU(name="relu_1"),
            tf.keras.layers.Dense(HIDDEN_DIMS[1], name="dense_2"),
            tf.keras.layers.ReLU(name="relu_2"),
            tf.keras.layers.Dense(NUM_CLASSES, name="logits"),
        ],
        name="tiny_mlp",
    )


def copy_pytorch_weights_to_keras(
    pytorch_model: torch.nn.Module,
    keras_model: Model,
) -> None:
    """把 TinyMLP 的三层 Linear 权重复制到对应的 Keras Dense 层。

    PyTorch ``Linear`` 权重布局为 ``[输出维, 输入维]``，Keras ``Dense`` kernel 布局为
    ``[输入维, 输出维]``，所以 kernel 必须转置；bias 在两个框架中都是一维输出向量，
    可以直接复制。函数只迁移参数，不修改任一模型的训练状态。
    """
    state_dict = pytorch_model.state_dict()
    # 显式列出映射而不是按层序号自动猜测，模型结构变化时能够尽早以缺失键或层名报错。
    layer_mapping = (
        ("network.0.weight", "network.0.bias", "dense_1"),
        ("network.2.weight", "network.2.bias", "dense_2"),
        ("network.4.weight", "network.4.bias", "logits"),
    )
    for weight_name, bias_name, keras_layer_name in layer_mapping:
        # detach 切断 autograd；cpu 保证 CUDA 检查点也能转换；copy 避免 NumPy 数组继续
        # 共享 PyTorch Tensor 内存。T 只作用于二维 kernel，bias 不需要转置。
        weight = state_dict[weight_name].detach().cpu().numpy().T.copy()
        bias = state_dict[bias_name].detach().cpu().numpy().copy()
        keras_model.get_layer(keras_layer_name).set_weights([weight, bias])


def _compare_logits(
    pytorch_model: torch.nn.Module,
    keras_model: Model,
    features: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, float | int]:
    """比较两种框架的 logits，并拒绝任何类别预测差异。

    数值等价和任务等价同时检查：``allclose`` 约束每个 logit 的误差，``argmax`` 一致率
    则确保微小浮点差异没有改变最终分类结果。只有两项都通过才返回统计摘要。
    """
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance 必须是有限非负数")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance 必须是有限非负数")

    # 两个框架接收同一块连续 float32 数据，避免输入 dtype 或内存布局差异干扰比较。
    feature_values = np.ascontiguousarray(features, dtype=np.float32)
    with torch.inference_mode():
        # 关闭 autograd，保证这里是纯推理，也减少不必要的中间张量和内存占用。
        pytorch_logits = (
            pytorch_model(torch.from_numpy(feature_values)).detach().cpu().numpy()
        )
    # UCI HAR 测试矩阵仅约 0.44 MiB。一次调用可避免 TensorFlow eager 模式在许多
    # 微小批次间重复调度，同时不会带来有意义的内存压力。
    keras_logits = keras_model(feature_values, training=False).numpy()
    absolute_error = np.abs(pytorch_logits - keras_logits)
    predictions_match = np.argmax(pytorch_logits, axis=1) == np.argmax(
        keras_logits, axis=1
    )
    prediction_agreement = float(np.mean(predictions_match))

    # atol 约束接近 0 的值，rtol 约束幅值较大的值；两者共同覆盖完整 logits 范围。
    if not np.allclose(
        pytorch_logits,
        keras_logits,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise RuntimeError(
            "PyTorch 与 Keras logits 未通过数值对齐："
            f"max_absolute_error={float(np.max(absolute_error)):.9g}"
        )
    # 即使 allclose 通过，决策边界附近的极小误差也可能翻转 argmax，因此单独强制检查。
    if not np.all(predictions_match):
        mismatch_count = int(np.count_nonzero(~predictions_match))
        raise RuntimeError(
            f"PyTorch 与 Keras 有 {mismatch_count} 条预测类别不一致"
        )

    return {
        "sample_count": int(features.shape[0]),
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "prediction_agreement": prediction_agreement,
    }


def _sha256(path: Path) -> str:
    """计算文件 SHA-256，用于关联量化和部署报告中的模型版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        # 分块读取，避免未来模型变大时一次性加载整个文件。
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save_json(path: Path, payload: dict[str, object]) -> None:
    """原子写入 UTF-8 JSON 报告。

    先写同目录临时文件，再用 ``os.replace`` 一次性提交。这样进程中断时最多遗留临时
    文件，不会让目标路径出现只写了一半、却看似有效的 JSON。
    """
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def export_checkpoint_to_keras(
    checkpoint_path: str | Path,
    processed_dir: str | Path,
    model_path: str | Path = DEFAULT_KERAS_MODEL_PATH,
    report_path: str | Path = DEFAULT_CONVERSION_REPORT_PATH,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
    overwrite: bool = False,
) -> KerasExportResult:
    """导出并验证与 PyTorch 最佳检查点等价的 Keras 模型。

    参数：
        checkpoint_path: ``training.train`` 生成的 PyTorch 最佳检查点。
        processed_dir: 含 ``test.npz`` 的标准化特征目录。
        model_path: 目标 Keras 文件，必须使用 ``.keras`` 后缀。
        report_path: 保存转换契约和数值验证结果的 JSON 路径。
        absolute_tolerance: logits 对齐使用的绝对容差。
        relative_tolerance: logits 对齐使用的相对容差。
        overwrite: 是否允许替换已有模型和报告。

    返回：
        包含输出路径、误差摘要和模型哈希的 :class:`KerasExportResult`。
    """
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值")

    # 统一将字符串参数转换为 Path；expanduser 允许调用方使用用户目录缩写。
    checkpoint_file = Path(checkpoint_path).expanduser()
    processed_source = Path(processed_dir).expanduser()
    destination = Path(model_path).expanduser()
    report_destination = Path(report_path).expanduser()
    if destination.suffix != ".keras":
        raise ValueError(f"Keras 模型路径必须使用 .keras 后缀：{destination}")
    # 默认拒绝覆盖，避免不同训练实验意外写到同一个默认输出路径。
    existing = [path for path in (destination, report_destination) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"转换输出已存在；如需覆盖请设置 overwrite=True：{joined}")

    # 阶段 1：加载固定 schema 的官方测试特征，以及经过严格契约校验的 PyTorch 模型。
    test_dataset = load_processed_split(processed_source / SPLIT_FILENAMES["test"])
    features = test_dataset.features.numpy()
    pytorch_model, checkpoint = load_best_model(
        checkpoint_file, device=torch.device("cpu")
    )
    # 阶段 2：先创建空的等价 Keras 网络，再迁移三层 Linear/Dense 参数。
    keras_model = build_keras_tiny_mlp()
    copy_pytorch_weights_to_keras(pytorch_model, keras_model)
    # 阶段 3：保存前先验证内存中的桥接模型，尽早发现转置或层映射错误。
    pre_save_metrics = _compare_logits(
        pytorch_model,
        keras_model,
        features,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    tf = _import_tensorflow()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    # 阶段 4：先保存到同目录临时 .keras 文件。Keras 根据后缀选择原生归档格式，因此
    # 临时文件也必须保留 .keras 后缀。
    temporary_model_path = destination.with_name(f".{destination.stem}.tmp.keras")
    temporary_model_path.unlink(missing_ok=True)
    try:
        # 这里只部署推理图，不保存未编译模型并不存在的优化器状态。
        keras_model.save(temporary_model_path, include_optimizer=False)
        # 必须验证“重新加载后的模型”，而不只验证保存前的内存对象。这样可以发现权重、
        # 自定义层或序列化配置丢失等问题。
        reloaded_model = tf.keras.models.load_model(
            temporary_model_path, compile=False
        )
        post_load_metrics = _compare_logits(
            pytorch_model,
            reloaded_model,
            features,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        # 两次对齐均通过后才原子替换最终模型；失败时 finally 只清理临时文件。
        os.replace(temporary_model_path, destination)
    finally:
        temporary_model_path.unlink(missing_ok=True)

    # 哈希取自落盘文件，TFLite/INT8 报告据此确认输入模型版本。
    model_sha256 = _sha256(destination)
    result = KerasExportResult(
        model_path=destination,
        report_path=report_destination,
        sample_count=int(post_load_metrics["sample_count"]),
        max_absolute_error=float(post_load_metrics["max_absolute_error"]),
        mean_absolute_error=float(post_load_metrics["mean_absolute_error"]),
        prediction_agreement=float(post_load_metrics["prediction_agreement"]),
        model_sha256=model_sha256,
    )
    # 报告同时保存数据契约、框架版本、权重映射和保存前/重载后两次结果，便于复现。
    report: dict[str, object] = {
        "format_version": KERAS_EXPORT_FORMAT_VERSION,
        "source_checkpoint": str(checkpoint_file.resolve()),
        "keras_model": str(destination.resolve()),
        "keras_model_sha256": model_sha256,
        "source_epoch": int(checkpoint["epoch"]),
        "model": {
            "name": "TinyMLP",
            "keras_name": keras_model.name,
            "structure": [FEATURE_DIM, *HIDDEN_DIMS, NUM_CLASSES],
            "layers": ["Dense", "ReLU", "Dense", "ReLU", "Dense"],
            "output": "logits",
        },
        "feature_names": list(FEATURE_NAMES),
        "activity_names": list(ACTIVITY_NAMES),
        "weight_mapping": {
            "network.0": "dense_1 (kernel transposed, bias copied)",
            "network.2": "dense_2 (kernel transposed, bias copied)",
            "network.4": "logits (kernel transposed, bias copied)",
        },
        "verification": {
            "split": "official_test",
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "before_save": pre_save_metrics,
            "after_reload": post_load_metrics,
        },
        "versions": {
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
        },
        "result": asdict(result),
    }
    report["result"]["model_path"] = str(destination.resolve())
    report["result"]["report_path"] = str(report_destination.resolve())
    _atomic_save_json(report_destination, report)
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/best_model.pt"),
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=DEFAULT_KERAS_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_CONVERSION_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有转换输出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数、执行导出，并打印适合人工检查的结果摘要。"""
    arguments = _build_argument_parser().parse_args(argv)
    result = export_checkpoint_to_keras(
        arguments.checkpoint,
        arguments.processed_dir,
        arguments.output,
        arguments.report,
        overwrite=arguments.overwrite,
    )
    print(f"Keras 模型：{result.model_path.resolve()}")
    print(f"转换报告：{result.report_path.resolve()}")
    print(
        "数值对齐："
        f"样本={result.sample_count}，"
        f"最大绝对误差={result.max_absolute_error:.9g}，"
        f"预测一致率={result.prediction_agreement:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONVERSION_REPORT_PATH",
    "DEFAULT_KERAS_MODEL_PATH",
    "KERAS_EXPORT_FORMAT_VERSION",
    "KerasExportResult",
    "build_keras_tiny_mlp",
    "copy_pytorch_weights_to_keras",
    "export_checkpoint_to_keras",
    "main",
]
