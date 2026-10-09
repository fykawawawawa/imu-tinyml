"""从官方测试集生成桌面 TFLM 一致性验证所需的 Golden Vector。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    REPOSITORY_ROOT / "models/exported/best_model_int8_baseline.tflite"
)
DEFAULT_TEST_DATA_PATH = REPOSITORY_ROOT / "data/processed/test.npz"
DEFAULT_HEADER_PATH = (
    REPOSITORY_ROOT / "firmware/include/inference/golden_vectors.h"
)
DEFAULT_SOURCE_PATH = (
    REPOSITORY_ROOT / "firmware/src/inference/golden_vectors.cc"
)

FEATURE_COUNT = 39
CLASS_COUNT = 6
ACTIVITY_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)


def _quantization_parameters(detail: dict[str, object]) -> tuple[float, int]:
    scale, zero_point = detail["quantization"]
    scale = float(scale)
    zero_point = int(zero_point)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"张量 {detail['name']} 缺少有效量化 scale")
    if not -128 <= zero_point <= 127:
        raise RuntimeError(f"张量 {detail['name']} 的 zero point 超出 INT8 范围")
    return scale, zero_point


def _run_tflite(
    model_path: Path,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, int], tuple[float, int]]:
    model_content = model_path.read_bytes()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError("Golden Vector 生成只支持 INT8 输入和输出模型")

    input_quantization = _quantization_parameters(input_detail)
    output_quantization = _quantization_parameters(output_detail)
    input_scale, input_zero_point = input_quantization

    values = np.ascontiguousarray(features, dtype=np.float32)
    unbounded = np.rint(values / input_scale) + input_zero_point
    quantized_input = np.clip(unbounded, -128, 127).astype(np.int8)

    interpreter.resize_tensor_input(
        int(input_detail["index"]),
        [values.shape[0], FEATURE_COUNT],
        strict=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(int(input_detail["index"]), quantized_input)
    interpreter.invoke()
    quantized_logits = interpreter.get_tensor(int(output_detail["index"]))
    return (
        quantized_input,
        np.asarray(quantized_logits, dtype=np.int8),
        input_quantization,
        output_quantization,
    )


def _format_float(value: np.float32) -> str:
    # 10 位有效数字足以无损往返 float32；科学计数法可确保生成合法浮点字面量。
    return f"{float(value):.9e}F"


def _format_rows(values: list[str], indentation: str, width: int) -> str:
    return "\n".join(
        indentation + ", ".join(values[offset : offset + width]) + ","
        for offset in range(0, len(values), width)
    )


def _render_header() -> str:
    return """\
#pragma once

#include <array>
#include <cstdint>

#include "inference/model_contract.h"

namespace imu::inference {

struct GoldenVector {
    int sample_index;
    int subject;
    int ground_truth_class;
    int expected_class;
    const char* activity_name;
    std::array<float, contract::kInputFeatureCount> normalized_features;
    std::array<std::int8_t, contract::kOutputClassCount> expected_quantized_logits;
};

extern const std::array<GoldenVector, contract::kOutputClassCount> kGoldenVectors;

}  // namespace imu::inference
"""


def _render_source(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    logits: np.ndarray,
    selected_indices: list[int],
    model_sha256: str,
    input_quantization: tuple[float, int],
    output_quantization: tuple[float, int],
) -> str:
    entries: list[str] = []
    for sample_index in selected_indices:
        expected_logits = logits[sample_index]
        expected_class = int(np.argmax(expected_logits))
        feature_rows = _format_rows(
            [_format_float(value) for value in features[sample_index]],
            "            ",
            6,
        )
        logit_row = ", ".join(str(int(value)) for value in expected_logits)
        entries.append(
            """\
    {
        SAMPLE_INDEX,
        SUBJECT,
        LABEL,
        EXPECTED_CLASS,
        "ACTIVITY",
        {{
FEATURE_ROWS
        }},
        {{LOGITS}},
    },"""
            .replace("SAMPLE_INDEX", str(sample_index))
            .replace("SUBJECT", str(int(subjects[sample_index])))
            .replace("LABEL", str(int(labels[sample_index])))
            .replace("EXPECTED_CLASS", str(expected_class))
            .replace("ACTIVITY", ACTIVITY_NAMES[int(labels[sample_index])])
            .replace("FEATURE_ROWS", feature_rows)
            .replace("LOGITS", logit_row)
        )

    input_scale, input_zero_point = input_quantization
    output_scale, output_zero_point = output_quantization
    return f"""\
#include "inference/golden_vectors.h"

// Generated by scripts/generate_tflm_golden_vectors.py.
// Model SHA256: {model_sha256}
// Input quantization: scale={input_scale:.17g}, zero_point={input_zero_point}
// Output quantization: scale={output_scale:.17g}, zero_point={output_zero_point}
namespace imu::inference {{

const std::array<GoldenVector, contract::kOutputClassCount> kGoldenVectors = {{{{
{chr(10).join(entries)}
}}}};

}}  // namespace imu::inference
"""


def generate(
    model_path: Path,
    test_data_path: Path,
    header_path: Path,
    source_path: Path,
) -> list[int]:
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 INT8 TFLite 模型：{model_path}")
    if not test_data_path.is_file():
        raise FileNotFoundError(f"找不到处理后测试集：{test_data_path}")

    with np.load(test_data_path, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["features"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        subjects = np.asarray(archive["subjects"], dtype=np.int64)

    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError(f"测试特征形状必须为 (N, {FEATURE_COUNT})：{features.shape}")
    if labels.shape != (features.shape[0],) or subjects.shape != labels.shape:
        raise ValueError("测试标签、受试者和特征样本数不一致")

    _, logits, input_quantization, output_quantization = _run_tflite(
        model_path,
        features,
    )
    predictions = np.argmax(logits, axis=1)

    selected_indices: list[int] = []
    for label in range(CLASS_COUNT):
        candidates = np.flatnonzero((labels == label) & (predictions == label))
        if candidates.size == 0:
            raise RuntimeError(f"类别 {label} 没有可用于 Golden Vector 的正确预测样本")
        selected_indices.append(int(candidates[0]))

    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(_render_header(), encoding="utf-8", newline="\n")
    source_path.write_text(
        _render_source(
            features,
            labels,
            subjects,
            logits,
            selected_indices,
            model_sha256,
            input_quantization,
            output_quantization,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return selected_indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA_PATH)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER_PATH)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    indices = generate(args.model, args.test_data, args.header, args.source)
    print(f"Generated: {args.header.resolve()}")
    print(f"Generated: {args.source.resolve()}")
    print("Selected test indices: " + ", ".join(str(index) for index in indices))


if __name__ == "__main__":
    main()
