"""将等价 Keras TinyMLP 导出为浮点 TensorFlow Lite 模型。

本模块不启用任何量化或优化选项，保留 float32 输入、权重、激活和 logits 输出。
导出后重新加载 TFLite FlatBuffer，并在官方测试特征上与 Keras 模型逐项比较。

该 FP32 TFLite 模型有两个用途：一是作为 INT8 量化的精度基线，二是隔离
“框架转换误差”和“量化误差”。因此本模块会严格检查 Keras 结构、TFLite 张量契约、
算子列表、逐项 logits 误差以及最终分类结果，不把“成功生成文件”等同于转换成功。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from feature_ref import FEATURE_DIM, FEATURE_NAMES
from models.mlp import HIDDEN_DIMS, NUM_CLASSES
from training.data import ACTIVITY_NAMES
from training.export_keras import _import_tensorflow
from training.preprocess import SPLIT_FILENAMES, load_processed_split

if TYPE_CHECKING:
    # 仅为静态类型检查导入 Model；运行时 TensorFlow 仍通过 export_keras 中的延迟导入
    # 函数加载，避免普通 PyTorch 命令被 TensorFlow 初始化开销拖慢。
    from keras import Model

# 浮点 TFLite 报告与 Keras 桥接报告使用各自版本号，便于独立演进。
TFLITE_FLOAT_EXPORT_FORMAT_VERSION = 1
# 默认路径构成明确的产物链：best_model.pt -> best_model.keras -> float32.tflite。
DEFAULT_KERAS_MODEL_PATH = Path("models/exported/best_model.keras")
DEFAULT_FLOAT_TFLITE_MODEL_PATH = Path("models/exported/best_model_float32.tflite")
DEFAULT_FLOAT_TFLITE_REPORT_PATH = Path(
    "models/exported/best_model_float32_tflite_report.json"
)


@dataclass(frozen=True)
class FloatTFLiteExportResult:
    """浮点 TFLite 导出文件及 Keras/TFLite 对齐结果摘要。

    输入/输出张量名称、动态形状和算子列表等详细信息保存在 JSON 报告中，避免调用方
    为读取常用误差指标而解析完整报告。
    """

    model_path: Path
    report_path: Path
    sample_count: int
    max_absolute_error: float
    mean_absolute_error: float
    prediction_agreement: float
    model_sha256: str


def _sha256(path: Path) -> str:
    """分块计算文件 SHA-256，用于串联 Keras 来源和 TFLite 产物。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """原子写入 TFLite FlatBuffer。

    ``flush`` 将 Python 缓冲区交给操作系统，``fsync`` 再请求系统落盘，最后用
    ``os.replace`` 提交目标文件。这样不会暴露部分写入的 FlatBuffer。
    """
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
    """通过同目录临时文件原子写入 UTF-8 JSON 报告。"""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _shape_tuple(shape: object) -> tuple[int | None, ...]:
    """把 Keras TensorShape 转换为便于严格比较和报错显示的 tuple。"""
    return tuple(None if dimension is None else int(dimension) for dimension in shape)


def _validate_keras_model_contract(model: Model) -> None:
    """确认输入模型仍是项目约定的 Dense/ReLU logits 网络。

    这里不仅检查输入输出维度，还检查层名称、层类型、参数量、输入输出个数与 dtype。
    这样即使调用方误传了“形状碰巧相同”的其他模型，也不会静默生成误导性产物。
    """
    # None 是动态批量维，39 和 6 分别是固定特征数与活动类别数。
    if _shape_tuple(model.input_shape) != (None, FEATURE_DIM):
        raise ValueError(
            f"Keras 输入形状必须为 (None, {FEATURE_DIM})，实际为 {model.input_shape}"
        )
    if _shape_tuple(model.output_shape) != (None, NUM_CLASSES):
        raise ValueError(
            f"Keras 输出形状必须为 (None, {NUM_CLASSES})，实际为 {model.output_shape}"
        )
    # 层名参与 PyTorch/Keras 权重映射和报告检查，因此与层类型一起严格校验。
    expected_layers = (
        ("dense_1", "Dense"),
        ("relu_1", "ReLU"),
        ("dense_2", "Dense"),
        ("relu_2", "ReLU"),
        ("logits", "Dense"),
    )
    actual_layers = tuple((layer.name, type(layer).__name__) for layer in model.layers)
    if actual_layers != expected_layers:
        raise ValueError(
            "Keras 层名称或类型与 TinyMLP 契约不一致："
            f"期望 {expected_layers!r}，实际 {actual_layers!r}"
        )
    # 参数量是结构校验的第二道防线，可捕获隐藏层宽度或 bias 配置的意外变化。
    if model.count_params() != 1910:
        raise ValueError(f"Keras 模型参数量必须为 1910，实际为 {model.count_params()}")
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError(
            "Keras 模型必须恰好包含一个输入和一个输出，实际为 "
            f"{len(model.inputs)}/{len(model.outputs)}"
        )
    # Keras 3 反序列化 Sequential 后，复数属性 inputs/outputs 比单数 input/output 稳定。
    input_dtype = str(model.inputs[0].dtype)
    output_dtype = str(model.outputs[0].dtype)
    if input_dtype != "float32" or output_dtype != "float32":
        raise ValueError(
            "Keras 模型输入输出必须为 float32，实际为 "
            f"{input_dtype}/{output_dtype}"
        )


def _create_float_interpreter(model_path: Path):
    """创建只使用内建 TFLite 算子、且单线程运行的解释器。

    禁用默认 delegate 可以避免 XNNPACK 等平台优化改变算子列表或浮点累计顺序，使
    Keras/TFLite 对齐结果在不同开发机上更容易复现。性能基准应另建专门流程。
    """
    tf = _import_tensorflow()
    return tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )


def _compare_keras_and_tflite(
    keras_model: Model,
    tflite_model_path: Path,
    features: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict[str, float | int], dict[str, object]]:
    """执行 TFLite FP32 推理并与 Keras logits 比较。

    返回两个字典：第一个是误差与预测一致率，第二个是部署端需要的输入/输出张量契约
    和算子清单。任何形状、dtype、逐项误差或分类结果不一致都会直接失败。
    """
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance 必须是有限非负数")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance 必须是有限非负数")

    # Keras 与 TFLite 必须接收完全相同的连续 float32 输入，避免隐式转换造成额外差异。
    values = np.ascontiguousarray(features, dtype=np.float32)
    keras_logits = keras_model(values, training=False).numpy()

    # 先读取未分配状态下的签名；此时 shape 通常是 [1, 39]，而 shape_signature 才保留
    # 模型声明的动态批量维 -1。
    interpreter = _create_float_interpreter(tflite_model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(
            "TFLite 模型必须恰好包含一个输入和一个输出，实际为 "
            f"{len(input_details)}/{len(output_details)}"
        )
    input_detail = input_details[0]
    output_detail = output_details[0]
    if input_detail["dtype"] != np.float32 or output_detail["dtype"] != np.float32:
        raise RuntimeError(
            "浮点 TFLite 输入输出必须为 float32，实际为 "
            f"{input_detail['dtype']}/{output_detail['dtype']}"
        )
    input_signature = tuple(int(value) for value in input_detail["shape_signature"])
    output_signature = tuple(int(value) for value in output_detail["shape_signature"])
    if input_signature != (-1, FEATURE_DIM) or output_signature != (-1, NUM_CLASSES):
        raise RuntimeError(
            "TFLite 动态形状契约不一致："
            f"input={input_signature}, output={output_signature}"
        )

    # 用完整测试集大小替换动态批量维。resize 之后必须重新 allocate_tensors，旧的张量
    # 索引描述也重新读取，避免依赖解释器实现中的索引稳定性。
    interpreter.resize_tensor_input(
        int(input_detail["index"]), [values.shape[0], FEATURE_DIM], strict=True
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(int(input_detail["index"]), values)
    interpreter.invoke()
    tflite_logits = interpreter.get_tensor(int(output_detail["index"]))

    # 同时检查连续数值输出和离散决策。前者适合定位转换误差，后者对应最终业务行为。
    absolute_error = np.abs(keras_logits - tflite_logits)
    predictions_match = np.argmax(keras_logits, axis=1) == np.argmax(
        tflite_logits, axis=1
    )
    if not np.allclose(
        keras_logits,
        tflite_logits,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise RuntimeError(
            "Keras 与浮点 TFLite logits 未通过数值对齐："
            f"max_absolute_error={float(np.max(absolute_error)):.9g}"
        )
    if not np.all(predictions_match):
        mismatch_count = int(np.count_nonzero(~predictions_match))
        raise RuntimeError(f"Keras 与浮点 TFLite 有 {mismatch_count} 条预测类别不一致")

    # TensorFlow 当前只通过私有诊断接口暴露算子明细，因此使用 getattr 兼容未来不存在
    # 该接口的版本；缺失时模型验证仍可完成，只是报告中的 operators 为空。
    get_ops_details = getattr(interpreter, "_get_ops_details", None)
    operators: list[str] = []
    if get_ops_details is not None:
        operators = [str(item["op_name"]) for item in get_ops_details()]
    verification: dict[str, float | int] = {
        "sample_count": int(values.shape[0]),
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "prediction_agreement": float(np.mean(predictions_match)),
    }
    # 记录部署端真正需要绑定的张量名、dtype 和动态形状。量化阶段还会在此基础上增加
    # scale/zero-point，但浮点基线应保持两端都是 float32。
    tensor_contract: dict[str, object] = {
        "input": {
            "name": str(input_detail["name"]),
            "dtype": np.dtype(input_detail["dtype"]).name,
            "shape_signature": list(input_signature),
        },
        "output": {
            "name": str(output_detail["name"]),
            "dtype": np.dtype(output_detail["dtype"]).name,
            "shape_signature": list(output_signature),
        },
        "operators": operators,
    }
    return verification, tensor_contract


def export_keras_to_float_tflite(
    keras_model_path: str | Path,
    processed_dir: str | Path,
    model_path: str | Path = DEFAULT_FLOAT_TFLITE_MODEL_PATH,
    report_path: str | Path = DEFAULT_FLOAT_TFLITE_REPORT_PATH,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
    overwrite: bool = False,
) -> FloatTFLiteExportResult:
    """把 Keras 桥接模型转换为经过全测试集验证的 FP32 TFLite 模型。

    参数：
        keras_model_path: 已通过 PyTorch/Keras 对齐的桥接模型路径。
        processed_dir: 含官方测试集 ``test.npz`` 的目录。
        model_path: 输出 FlatBuffer 路径，必须使用 ``.tflite`` 后缀。
        report_path: 输出转换与张量契约报告的 JSON 路径。
        absolute_tolerance: Keras/TFLite logits 的绝对容差。
        relative_tolerance: Keras/TFLite logits 的相对容差。
        overwrite: 是否允许在验证成功后替换已有产物。

    返回：
        输出路径、数值误差、预测一致率和哈希摘要。
    """
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值")

    # 先规范化路径并完成所有只读校验，避免转换运行到一半才发现输出参数错误。
    keras_source = Path(keras_model_path).expanduser()
    processed_source = Path(processed_dir).expanduser()
    destination = Path(model_path).expanduser()
    report_destination = Path(report_path).expanduser()
    if not keras_source.is_file():
        raise FileNotFoundError(f"找不到 Keras 模型：{keras_source}")
    if destination.suffix != ".tflite":
        raise ValueError(f"TFLite 模型路径必须使用 .tflite 后缀：{destination}")
    # 默认拒绝覆盖，显式 --overwrite 才允许更新已有基线模型。
    existing = [path for path in (destination, report_destination) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"转换输出已存在；如需覆盖请设置 overwrite=True：{joined}")

    # 阶段 1：加载并验证桥接模型。compile=False 避免恢复训练损失、优化器等无关状态。
    tf = _import_tensorflow()
    keras_model = tf.keras.models.load_model(keras_source, compile=False)
    _validate_keras_model_contract(keras_model)
    # 官方测试集只用于转换后的最终一致性检查，不参与模型训练或参数拟合。
    test_dataset = load_processed_split(processed_source / SPLIT_FILENAMES["test"])

    # 阶段 2：保持 Converter 默认 FP32 路径。这里故意不设置 optimizations、
    # representative_dataset 或 inference_input/output_type，避免提前引入任何量化行为。
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_payload = converter.convert()
    if not isinstance(tflite_payload, bytes) or not tflite_payload:
        raise RuntimeError("TensorFlow Lite Converter 未返回有效 FlatBuffer")

    destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    # 阶段 3：先把候选 FlatBuffer 写到隐藏验证路径。即使指定 overwrite，只有候选模型
    # 完整通过测试集验证后才会替换旧模型，防止失败转换破坏已知有效产物。
    verification_path = destination.with_name(f".{destination.name}.verify")
    verification_path.unlink(missing_ok=True)
    try:
        _atomic_write_bytes(verification_path, tflite_payload)
        verification, tensor_contract = _compare_keras_and_tflite(
            keras_model,
            verification_path,
            test_dataset.features.numpy(),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        # 同目录 os.replace 是原子提交；观察者只会看到旧的完整模型或新的完整模型。
        os.replace(verification_path, destination)
    finally:
        # 验证成功前不替换已有模型；失败时只清理本次生成的候选文件。
        verification_path.unlink(missing_ok=True)

    # 阶段 4：对最终文件计算哈希并生成报告，供 INT8 转换追踪来源。
    model_sha256 = _sha256(destination)
    result = FloatTFLiteExportResult(
        model_path=destination,
        report_path=report_destination,
        sample_count=int(verification["sample_count"]),
        max_absolute_error=float(verification["max_absolute_error"]),
        mean_absolute_error=float(verification["mean_absolute_error"]),
        prediction_agreement=float(verification["prediction_agreement"]),
        model_sha256=model_sha256,
    )
    # 报告中的特征/标签顺序是部署契约的一部分，不能只依赖模型张量的 39/6 维形状。
    report: dict[str, object] = {
        "format_version": TFLITE_FLOAT_EXPORT_FORMAT_VERSION,
        "conversion": "keras_float32_to_tflite_float32",
        "source_keras_model": str(keras_source.resolve()),
        "source_keras_sha256": _sha256(keras_source),
        "tflite_model": str(destination.resolve()),
        "tflite_model_sha256": model_sha256,
        "model": {
            "structure": [FEATURE_DIM, *HIDDEN_DIMS, NUM_CLASSES],
            "layers": ["Dense", "ReLU", "Dense", "ReLU", "Dense"],
            "parameter_count": int(keras_model.count_params()),
            "numeric_format": "float32",
            "output": "logits",
        },
        "feature_names": list(FEATURE_NAMES),
        "activity_names": list(ACTIVITY_NAMES),
        "tensor_contract": tensor_contract,
        "verification": {
            "split": "official_test",
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            **verification,
        },
        "versions": {
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
        },
        "result": {
            "model_path": str(destination.resolve()),
            "report_path": str(report_destination.resolve()),
            "sample_count": result.sample_count,
            "max_absolute_error": result.max_absolute_error,
            "mean_absolute_error": result.mean_absolute_error,
            "prediction_agreement": result.prediction_agreement,
            "model_sha256": result.model_sha256,
        },
    }
    _atomic_save_json(report_destination, report)
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras-model", type=Path, default=DEFAULT_KERAS_MODEL_PATH)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=DEFAULT_FLOAT_TFLITE_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_FLOAT_TFLITE_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有转换输出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数、执行转换，并打印核心验证摘要。"""
    arguments = _build_argument_parser().parse_args(argv)
    result = export_keras_to_float_tflite(
        arguments.keras_model,
        arguments.processed_dir,
        arguments.output,
        arguments.report,
        overwrite=arguments.overwrite,
    )
    print(f"浮点 TFLite 模型：{result.model_path.resolve()}")
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
    "DEFAULT_FLOAT_TFLITE_MODEL_PATH",
    "DEFAULT_FLOAT_TFLITE_REPORT_PATH",
    "DEFAULT_KERAS_MODEL_PATH",
    "FloatTFLiteExportResult",
    "TFLITE_FLOAT_EXPORT_FORMAT_VERSION",
    "export_keras_to_float_tflite",
    "main",
]
