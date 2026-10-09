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
