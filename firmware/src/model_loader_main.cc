#include <array>
#include <cstddef>
#include <iostream>

#include "inference/golden_vectors.h"
#include "inference/inference_runner.h"
#include "inference/model_contract.h"
#include "inference/tflm_interpreter.h"

int main() {
    // 桌面入口验证完整链路：加载并校验内嵌 FlatBuffer、注册算子、从静态 arena 分配
    // 张量并核对契约；初始化成功后还会用六类 Golden Vector 执行真实 Invoke()。
    const imu::inference::InterpreterInitStatus status =
        imu::inference::InitializeInterpreter();

    // 状态消息由解释器模块统一维护，避免入口程序复制错误判断逻辑。返回非零退出码后，
    // Make、CTest 或 CI 都能把模型不兼容、算子缺失和 arena 不足等问题识别为失败。
    if (status != imu::inference::InterpreterInitStatus::kOk) {
        std::cerr << "Failed to initialize MicroInterpreter: "
                  << imu::inference::InterpreterInitStatusMessage(status)
                  << '\n';
        return 1;
    }

    // InitializeInterpreter() 成功后，这三个访问器都应返回由解释器模块持有的有效对象。
    // 再做一次空值检查可以保护桌面测试入口，防止未来接口实现变化后出现静默崩溃。
    tflite::MicroInterpreter* interpreter =
        imu::inference::GetInterpreter();
    TfLiteTensor* input = imu::inference::GetInputTensor();
    TfLiteTensor* output = imu::inference::GetOutputTensor();
    if (interpreter == nullptr || input == nullptr || output == nullptr) {
        std::cerr << "MicroInterpreter reported success but returned null objects.\n";
        return 1;
    }

    // shape、类型、字节数及量化参数已经在解释器模块中严格验证。这里输出实际张量值，
    // 使运行日志能够直接证明当前编译进程序的模型与 model_contract.h 保持一致。
    std::cout << "MicroInterpreter initialized successfully.\n"
              << "Tensor arena reserved: "
              << imu::inference::contract::kTensorArenaSize << " bytes\n"
              << "Inputs: " << interpreter->inputs_size() << '\n'
              << "Input tensor: int8 [" << input->dims->data[0] << ", "
              << input->dims->data[1] << "]"
              << ", bytes=" << input->bytes
              << ", scale=" << input->params.scale
              << ", zero_point=" << input->params.zero_point << '\n'
              << "Outputs: " << interpreter->outputs_size() << '\n'
              << "Output tensor: int8 [" << output->dims->data[0] << ", "
              << output->dims->data[1] << "]"
              << ", bytes=" << output->bytes
              << ", scale=" << output->params.scale
              << ", zero_point=" << output->params.zero_point << '\n';

    // Golden Vector 来自同一 INT8 FlatBuffer 的 Python TFLite 推理。每个真实测试样本
    // 覆盖一个活动类别；TFLM 类别必须完全一致，六个整数输出允许最大 1 LSB 的平台差异。
    constexpr int kMaximumLogitDifference = 1;
    std::size_t passed_count = 0;
    for (const imu::inference::GoldenVector& golden :
         imu::inference::kGoldenVectors) {
        imu::inference::InferenceResult inference_result;
        const imu::inference::InferenceStatus inference_status =
            imu::inference::RunInference(
                golden.normalized_features.data(),
                golden.normalized_features.size(),
                &inference_result);
        if (inference_status != imu::inference::InferenceStatus::kOk) {
            std::cerr << "Golden sample " << golden.sample_index
                      << " inference failed: "
                      << imu::inference::InferenceStatusMessage(inference_status)
                      << '\n';
            return 1;
        }

        int maximum_difference = 0;
        bool logits_match = true;
        for (std::size_t index = 0;
             index < golden.expected_quantized_logits.size();
             ++index) {
            const int actual = static_cast<int>(
                inference_result.quantized_logits[index]);
            const int expected = static_cast<int>(
                golden.expected_quantized_logits[index]);
            const int difference = actual >= expected
                                       ? actual - expected
                                       : expected - actual;
            if (difference > maximum_difference) {
                maximum_difference = difference;
            }
            if (difference > kMaximumLogitDifference) {
                logits_match = false;
            }
        }

        const bool class_matches =
            inference_result.predicted_class == golden.expected_class;
        const bool passed = class_matches && logits_match;
        std::cout << "Golden sample=" << golden.sample_index
                  << " subject=" << golden.subject
                  << " activity=" << golden.activity_name
                  << " expected_class=" << golden.expected_class
                  << " actual_class=" << inference_result.predicted_class
                  << " max_logit_diff=" << maximum_difference
                  << " result=" << (passed ? "PASS" : "FAIL") << '\n';

        if (!passed) {
            std::cerr << "Expected logits:";
            for (const std::int8_t value :
                 golden.expected_quantized_logits) {
                std::cerr << ' ' << static_cast<int>(value);
            }
            std::cerr << "\nActual logits:";
            for (const std::int8_t value :
                 inference_result.quantized_logits) {
                std::cerr << ' ' << static_cast<int>(value);
            }
            std::cerr << '\n';
            return 1;
        }
        ++passed_count;
    }

    std::cout << "Golden Vector validation passed: " << passed_count << '/'
              << imu::inference::kGoldenVectors.size() << " samples.\n";

    return 0;
}
