#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "inference/model_contract.h"

namespace imu::inference {

struct InferenceResult {
    int predicted_class = -1;
    std::array<std::int8_t, contract::kOutputClassCount> quantized_logits{};
    std::array<float, contract::kOutputClassCount> dequantized_logits{};
};

enum class InferenceStatus {
    kOk,
    kInterpreterNotInitialized,
    kNullArgument,
    kInvalidFeatureCount,
    kNonFiniteFeature,
    kTensorDataUnavailable,
    kInvokeFailed,
};

// Quantizes one normalized 39-feature sample, invokes TFLM, and returns both
// INT8 and dequantized logits plus their argmax class. The feature array must
// already use the same training-set standardization as the exported model.
InferenceStatus RunInference(
    const float* normalized_features,
    std::size_t feature_count,
    InferenceResult* result);

const char* InferenceStatusMessage(InferenceStatus status);

}  // namespace imu::inference
