#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "inference/golden_vectors.h"
#include "inference/inference_runner.h"
#include "inference/model_contract.h"
#include "inference/tflm_interpreter.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWarmupRuns = 1000;
constexpr std::size_t kMeasuredRuns = 50000;

struct LatencySummary {
    double minimum_us;
    double mean_us;
    double p50_us;
    double p95_us;
    double p99_us;
    double maximum_us;
};

double Percentile(
    const std::vector<double>& sorted_microseconds,
    double fraction) {
    const double rank = std::ceil(
        fraction * static_cast<double>(sorted_microseconds.size()));
    const std::size_t index = static_cast<std::size_t>(rank) - 1U;
    return sorted_microseconds[index];
}

LatencySummary Summarize(std::vector<double> microseconds) {
    std::sort(microseconds.begin(), microseconds.end());
    const double total = std::accumulate(
        microseconds.begin(), microseconds.end(), 0.0);
    return {
        microseconds.front(),
        total / static_cast<double>(microseconds.size()),
        Percentile(microseconds, 0.50),
        Percentile(microseconds, 0.95),
        Percentile(microseconds, 0.99),
        microseconds.back(),
    };
}

void PrintSummary(const char* metric, const LatencySummary& summary) {
    std::cout << metric
              << " min_us=" << summary.minimum_us
              << " mean_us=" << summary.mean_us
              << " p50_us=" << summary.p50_us
              << " p95_us=" << summary.p95_us
              << " p99_us=" << summary.p99_us
              << " max_us=" << summary.maximum_us << '\n';
}

std::vector<double> MeasureClockOverhead() {
    std::vector<double> durations;
    durations.reserve(kMeasuredRuns);
    for (std::size_t index = 0; index < kMeasuredRuns; ++index) {
        const auto start = Clock::now();
        const auto end = Clock::now();
        durations.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
    }
    return durations;
}

}  // namespace

int main() {
    static_assert(Clock::is_steady, "latency benchmark requires a steady clock");

    const auto initialization_start = Clock::now();
    const imu::inference::InterpreterInitStatus initialization_status =
        imu::inference::InitializeInterpreter();
    const auto initialization_end = Clock::now();
    if (initialization_status !=
        imu::inference::InterpreterInitStatus::kOk) {
        std::cerr << "MicroInterpreter initialization failed: "
                  << imu::inference::InterpreterInitStatusMessage(
                         initialization_status)
                  << '\n';
        return 1;
    }

    tflite::MicroInterpreter* interpreter =
        imu::inference::GetInterpreter();
    TfLiteTensor* input = imu::inference::GetInputTensor();
    TfLiteTensor* output = imu::inference::GetOutputTensor();
    if (interpreter == nullptr || input == nullptr || output == nullptr ||
        input->data.int8 == nullptr || output->data.int8 == nullptr) {
        std::cerr << "Initialized interpreter did not expose usable tensors.\n";
        return 1;
    }

    using QuantizedInput =
        std::array<std::int8_t, imu::inference::contract::kInputFeatureCount>;
    std::array<QuantizedInput, imu::inference::contract::kOutputClassCount>
        quantized_inputs{};

    // Reuse the production quantizer to prepare one input per class. This work
    // happens before the timed Invoke-only loop, so that metric contains only
    // MicroInterpreter::Invoke().
    imu::inference::InferenceResult inference_result;
    for (std::size_t index = 0;
         index < imu::inference::kGoldenVectors.size();
         ++index) {
        const auto& features =
            imu::inference::kGoldenVectors[index].normalized_features;
        const imu::inference::InferenceStatus status =
            imu::inference::RunInference(
                features.data(), features.size(), &inference_result);
        if (status != imu::inference::InferenceStatus::kOk) {
            std::cerr << "Failed to prepare benchmark input: "
                      << imu::inference::InferenceStatusMessage(status) << '\n';
            return 1;
        }
        std::memcpy(
            quantized_inputs[index].data(),
            input->data.int8,
            quantized_inputs[index].size());
    }

    std::int64_t checksum = 0;

    for (std::size_t index = 0; index < kWarmupRuns; ++index) {
        const QuantizedInput& sample =
            quantized_inputs[index % quantized_inputs.size()];
        std::memcpy(input->data.int8, sample.data(), sample.size());
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Invoke-only warmup failed.\n";
            return 1;
        }
        checksum += static_cast<std::int64_t>(output->data.int8[0]);
    }

    std::vector<double> invoke_durations;
    invoke_durations.reserve(kMeasuredRuns);
    for (std::size_t index = 0; index < kMeasuredRuns; ++index) {
        const QuantizedInput& sample =
            quantized_inputs[index % quantized_inputs.size()];
        std::memcpy(input->data.int8, sample.data(), sample.size());

        const auto start = Clock::now();
        const TfLiteStatus status = interpreter->Invoke();
        const auto end = Clock::now();
        if (status != kTfLiteOk) {
            std::cerr << "Timed Invoke failed.\n";
            return 1;
        }
        invoke_durations.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
        checksum += static_cast<std::int64_t>(output->data.int8[0]);
    }

    for (std::size_t index = 0; index < kWarmupRuns; ++index) {
        const auto& features = imu::inference::kGoldenVectors[
            index % imu::inference::kGoldenVectors.size()]
                                   .normalized_features;
        const imu::inference::InferenceStatus status =
            imu::inference::RunInference(
                features.data(), features.size(), &inference_result);
        if (status != imu::inference::InferenceStatus::kOk) {
            std::cerr << "End-to-end warmup failed.\n";
            return 1;
        }
        checksum += inference_result.predicted_class;
    }

    std::vector<double> end_to_end_durations;
    end_to_end_durations.reserve(kMeasuredRuns);
    for (std::size_t index = 0; index < kMeasuredRuns; ++index) {
        const auto& features = imu::inference::kGoldenVectors[
            index % imu::inference::kGoldenVectors.size()]
                                   .normalized_features;

        const auto start = Clock::now();
        const imu::inference::InferenceStatus status =
            imu::inference::RunInference(
                features.data(), features.size(), &inference_result);
        const auto end = Clock::now();
        if (status != imu::inference::InferenceStatus::kOk) {
            std::cerr << "Timed end-to-end inference failed.\n";
            return 1;
        }
        end_to_end_durations.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
        checksum += inference_result.predicted_class;
    }

    const double initialization_us =
        std::chrono::duration<double, std::micro>(
            initialization_end - initialization_start)
            .count();

    std::cout << std::fixed << std::setprecision(3)
              << "benchmark warmup_runs=" << kWarmupRuns
              << " measured_runs=" << kMeasuredRuns << '\n'
              << "initialization latency_us=" << initialization_us << '\n';
    PrintSummary("clock_overhead", Summarize(MeasureClockOverhead()));
    PrintSummary("invoke_only", Summarize(std::move(invoke_durations)));
    PrintSummary("end_to_end", Summarize(std::move(end_to_end_durations)));
    std::cout << "checksum=" << checksum << '\n';

    return 0;
}
