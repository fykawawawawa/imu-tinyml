#pragma once

#include <cstddef>

namespace imu::inference::contract {

inline constexpr int kInputBatchSize = 1;
inline constexpr int kInputFeatureCount = 39;
inline constexpr int kOutputBatchSize = 1;
inline constexpr int kOutputClassCount = 6;

inline constexpr float kInputScale = 0.1474212110042572F;
inline constexpr int kInputZeroPoint = 31;

inline constexpr float kOutputScale = 0.23944978415966034F;
inline constexpr int kOutputZeroPoint = 19;

// RecordingMicroInterpreter measured 1,888 bytes for this exact model and TFLM
// revision. Reserve 2,304 bytes: 416 bytes (about 22%) of aligned headroom for
// minor allocator/platform variation. Re-profile whenever the model or TFLM changes.
inline constexpr std::size_t kTensorArenaSize = 2304;

}  // namespace imu::inference::contract
