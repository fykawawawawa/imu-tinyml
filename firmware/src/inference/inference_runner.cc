#include "inference/inference_runner.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "inference/tflm_interpreter.h"

namespace imu::inference {
namespace {

std::int8_t QuantizeFeature(float feature, const TfLiteTensor& input) {
    // q = round(real / scale) + zero_point。先在浮点域判断两端饱和，可以避免极端
    // 有限输入传给 lround 时超出 long 的范围；正常范围内再执行与 TFLite 相同的舍入。
    const float scaled = feature / input.params.scale;
    const float minimum_centered =
        static_cast<float>(std::numeric_limits<std::int8_t>::min() -
                           input.params.zero_point);
    const float maximum_centered =
        static_cast<float>(std::numeric_limits<std::int8_t>::max() -
                           input.params.zero_point);
    // clip截断
    if (scaled <= minimum_centered) {
        return std::numeric_limits<std::int8_t>::min();
    }
    if (scaled >= maximum_centered) {
        return std::numeric_limits<std::int8_t>::max();
    }

    // Python 基准使用 np.rint（默认 ties-to-even）。nearbyint 在默认舍入模式下采用同一
    // 规则，确保恰好位于半量化步长上的输入也能与 Golden Vector 保持一致。
    const long rounded = static_cast<long>(std::nearbyint(scaled));
    return static_cast<std::int8_t>(
        rounded + static_cast<long>(input.params.zero_point));
}

float DequantizeLogit(std::int8_t value, const TfLiteTensor& output) {
    const int centered = static_cast<int>(value) - output.params.zero_point;
    return static_cast<float>(centered) * output.params.scale;
}

}  // namespace

InferenceStatus RunInference(
    const float* normalized_features,
    std::size_t feature_count,
    InferenceResult* result) {
    if (normalized_features == nullptr || result == nullptr) {
        return InferenceStatus::kNullArgument;
    }
    if (feature_count !=
        static_cast<std::size_t>(contract::kInputFeatureCount)) {
        return InferenceStatus::kInvalidFeatureCount;
    }

    // 在写入输入 tensor 前先完成整组校验，避免错误样本只覆盖一部分旧输入数据。
    for (std::size_t index = 0; index < feature_count; ++index) {
        if (!std::isfinite(normalized_features[index])) {
            return InferenceStatus::kNonFiniteFeature;
        }
    }

    tflite::MicroInterpreter* interpreter = GetInterpreter();
    TfLiteTensor* input = GetInputTensor();
    TfLiteTensor* output = GetOutputTensor();
    if (interpreter == nullptr || input == nullptr || output == nullptr) {
        return InferenceStatus::kInterpreterNotInitialized;
    }
    if (input->data.int8 == nullptr || output->data.int8 == nullptr) {
        return InferenceStatus::kTensorDataUnavailable;
    }

    // 输入 tensor 的 shape、类型和量化参数已在 InitializeInterpreter() 中验证。
    // 此处只负责逐特征量化并写入连续的 [1, 39] INT8 缓冲区。
    for (std::size_t index = 0; index < feature_count; ++index) {
        input->data.int8[index] =
            QuantizeFeature(normalized_features[index], *input);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        return InferenceStatus::kInvokeFailed;
    }

    // 所有输出共享同一组 scale/zero point，所以可以直接在 INT8 域取 argmax；反量化
    // logits 仅供日志、误差分析和后续 Golden Vector 对比使用。
    int predicted_class = 0;
    for (int index = 0; index < contract::kOutputClassCount; ++index) {
        const std::int8_t quantized = output->data.int8[index];
        result->quantized_logits[static_cast<std::size_t>(index)] = quantized;
        result->dequantized_logits[static_cast<std::size_t>(index)] =
            DequantizeLogit(quantized, *output);

        if (index > 0 &&
            quantized > output->data.int8[predicted_class]) {
            predicted_class = index;
        }
    }
    result->predicted_class = predicted_class;
    return InferenceStatus::kOk;
}

const char* InferenceStatusMessage(InferenceStatus status) {
    switch (status) {
        case InferenceStatus::kOk:
            return "inference completed successfully";
        case InferenceStatus::kInterpreterNotInitialized:
            return "MicroInterpreter is not initialized";
        case InferenceStatus::kNullArgument:
            return "feature or result pointer is null";
        case InferenceStatus::kInvalidFeatureCount:
            return "feature count does not match the model input";
        case InferenceStatus::kNonFiniteFeature:
            return "feature array contains NaN or infinity";
        case InferenceStatus::kTensorDataUnavailable:
            return "input or output tensor data is unavailable";
        case InferenceStatus::kInvokeFailed:
            return "MicroInterpreter Invoke() failed";
    }
    return "unknown inference status";
}

}  // namespace imu::inference
