#include "inference/tflm_interpreter.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>

#include "inference/model_contract.h"
#include "inference/model_loader.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace imu::inference {
namespace {

// 当前模型虽然包含三个全连接节点，但它们都使用同一种 FULLY_CONNECTED 算子，因此
// resolver 只需要一个注册槽。这样可以避免 AllOpsResolver 把未使用算子带入最终程序。
using ModelOpResolver = tflite::MicroMutableOpResolver<1>;

// 算子注册器、arena 和解释器都是静态对象，生命周期覆盖整个进程。
ModelOpResolver g_op_resolver;

// MicroInterpreter 不负责 tensor arena 的所有权；arena 必须比解释器存活得更久。
// 将其定义为本翻译单元中的静态对象，可避免把数组放到函数调用栈上，并防止其他模块
// 直接修改。16 字节对齐也满足桌面平台及后续嵌入式平台的常见对齐要求。
alignas(16) std::uint8_t g_tensor_arena[contract::kTensorArenaSize] = {};

// optional 只提供静态对象内部的延迟构造能力，不会为 MicroInterpreter 分配堆内存。
// 只有模型和算子 resolver 准备好后才构造解释器，避免使用无效模型指针。
std::optional<tflite::MicroInterpreter> g_interpreter;

// TfLiteTensor* 由 MicroInterpreter 内部持有，生命周期与解释器一致。初始化成功后
// 公开给外部访问，便于推理时直接写入输入张量
TfLiteTensor* g_input_tensor = nullptr;
TfLiteTensor* g_output_tensor = nullptr;

// 初始化只允许执行一次。重复注册同一个算子可能失败，因此后续调用直接返回第一次的
// 结果；若第一次失败，需要修复配置并重新启动进程。
bool g_initialization_attempted = false;
bool g_initialization_succeeded = false;
InterpreterInitStatus g_initialization_result =
    InterpreterInitStatus::kModelLoadFailed;

// 量化 scale 存储为 float。使用小容差比较可避免因十进制常量转换造成无意义的不相等，
// 同时该容差足够严格，能够发现重新量化后真实发生变化的模型契约。
bool NearlyEqual(float actual, float expected) {
    constexpr float kScaleTolerance = 1.0e-7F;
    return std::fabs(actual - expected) <= kScaleTolerance;
}

bool HasExpectedShape(
    const TfLiteTensor* tensor,
    int expected_batch_size,
    int expected_element_count) {
    return tensor != nullptr && tensor->dims != nullptr &&
           tensor->dims->size == 2 &&
           tensor->dims->data[0] == expected_batch_size &&
           tensor->dims->data[1] == expected_element_count;
}

// TFLM 在构造运行时 TfLiteTensor 时会稳定提供 params.scale 和 params.zero_point，
// 推理时的量化/反量化也使用这两个值。完整的 tensor->quantization affine 元数据是
// 可选信息，在精简的 MicroInterpreter 构建中可能不会按桌面 TFLite 的形式保留，因而
// 不应把它作为初始化成功的必要条件。这里仍严格比较实际参与推理的 scale 和 zero point。
bool HasExpectedQuantization(
    const TfLiteTensor* tensor,
    float expected_scale,
    int expected_zero_point) {
    return tensor != nullptr &&
           NearlyEqual(tensor->params.scale, expected_scale) &&
           tensor->params.zero_point == expected_zero_point;
}

bool ValidateInputTensor(const TfLiteTensor* input) {
    constexpr std::size_t kExpectedBytes =
        contract::kInputBatchSize * contract::kInputFeatureCount *
        sizeof(std::int8_t);

    return input != nullptr && input->type == kTfLiteInt8 &&
           HasExpectedShape(
               input,
               contract::kInputBatchSize,
               contract::kInputFeatureCount) &&
           input->bytes == kExpectedBytes &&
           HasExpectedQuantization(
               input,
               contract::kInputScale,
               contract::kInputZeroPoint);
}

bool ValidateOutputTensor(const TfLiteTensor* output) {
    constexpr std::size_t kExpectedBytes =
        contract::kOutputBatchSize * contract::kOutputClassCount *
        sizeof(std::int8_t);

    return output != nullptr && output->type == kTfLiteInt8 &&
           HasExpectedShape(
               output,
               contract::kOutputBatchSize,
               contract::kOutputClassCount) &&
           output->bytes == kExpectedBytes &&
           HasExpectedQuantization(
               output,
               contract::kOutputScale,
               contract::kOutputZeroPoint);
}

// 初始化失败时输出运行时实际看到的 tensor 契约，便于区分 shape、类型、字节数和
// 量化参数问题。该函数只在桌面诊断失败路径调用，不参与正常推理热路径。
void PrintTensorContract(const char* name, const TfLiteTensor* tensor) {
    if (tensor == nullptr) {
        std::fprintf(stderr, "%s tensor is null\n", name);
        return;
    }

    std::fprintf(
        stderr,
        "%s tensor: type=%d, bytes=%zu, scale=%.9g, zero_point=%d, shape=[",
        name,
        static_cast<int>(tensor->type),
        tensor->bytes,
        static_cast<double>(tensor->params.scale),
        tensor->params.zero_point);
    if (tensor->dims == nullptr) {
        std::fprintf(stderr, "null");
    } else {
        for (int index = 0; index < tensor->dims->size; ++index) {
            std::fprintf(
                stderr,
                "%s%d",
                index == 0 ? "" : ", ",
                tensor->dims->data[index]);
        }
    }
    std::fprintf(stderr, "]\n");
}

InterpreterInitStatus FinishInitialization(InterpreterInitStatus status) {
    g_initialization_result = status;
    g_initialization_succeeded = status == InterpreterInitStatus::kOk;
    return status;
}

}  // namespace

// InterpreterInitStatus来自tflm_interpreter.h，外部调用者可以根据返回值判断初始化是否成功，并在失败时输出诊断信息。
InterpreterInitStatus InitializeInterpreter() {
    if (g_initialization_attempted) {
        return g_initialization_result;
    }
    g_initialization_attempted = true;

    // 复用前一阶段的模型加载器。它已经校验了 TFL3 标识、完整 FlatBuffer 结构和
    // TFLITE_SCHEMA_VERSION，因此这里只接受完全成功的加载结果。
    const ModelLoadResult load_result = LoadEmbeddedModel();
    if (load_result.status != ModelLoadStatus::kOk ||
        load_result.model == nullptr) {
        return FinishInitialization(InterpreterInitStatus::kModelLoadFailed);
    }

    // 仅注册模型实际使用的算子。本模型中的 ReLU 已融合进 FULLY_CONNECTED 节点，
    // FlatBuffer 中没有独立的 RELU 算子，因此不需要额外注册。
    if (g_op_resolver.AddFullyConnected() != kTfLiteOk) {
        return FinishInitialization(
            InterpreterInitStatus::kOperatorRegistrationFailed);
    }

    // MicroInterpreter 只保存 model、resolver 和 arena 的引用，不复制这些对象。
    // 三者均为静态或编译期模型数据，因此生命周期覆盖整个解释器使用期。
    g_interpreter.emplace(
        load_result.model,
        g_op_resolver,
        g_tensor_arena,
        sizeof(g_tensor_arena));

    // AllocateTensors 会解析图、为持久数据及中间张量规划 arena。此处失败最常见的
    // 原因是 arena 太小，也可能表示模型包含未注册或当前 TFLM 不支持的算子。
    // arena是TFLM用于模型推理的静态内存区域，必须在解释器初始化时分配。
    // AllocateTensors()会根据模型的需求分配张量所需的内存，如果arena不足，则会返回错误。
    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        return FinishInitialization(
            InterpreterInitStatus::kTensorAllocationFailed);
    }

    // 当前模型契约严格规定单输入、单输出；先检查数量，避免访问不存在的索引。
    if (g_interpreter->inputs_size() != 1 ||
        g_interpreter->outputs_size() != 1) {
        return FinishInitialization(
            InterpreterInitStatus::kTensorCountMismatch);
    }

    TfLiteTensor* input = g_interpreter->input(0);
    TfLiteTensor* output = g_interpreter->output(0);

    if (!ValidateInputTensor(input)) {
        PrintTensorContract("Input", input);
        return FinishInitialization(
            InterpreterInitStatus::kInputContractMismatch);
    }
    if (!ValidateOutputTensor(output)) {
        PrintTensorContract("Output", output);
        return FinishInitialization(
            InterpreterInitStatus::kOutputContractMismatch);
    }

    // 只有所有检查都成功后才公开 tensor 指针，避免调用方误用部分初始化的解释器。
    g_input_tensor = input;
    g_output_tensor = output;
    return FinishInitialization(InterpreterInitStatus::kOk);
}

const char* InterpreterInitStatusMessage(InterpreterInitStatus status) {
    switch (status) {
        case InterpreterInitStatus::kOk:
            return "MicroInterpreter initialized successfully";
        case InterpreterInitStatus::kModelLoadFailed:
            return "embedded model loading or schema validation failed";
        case InterpreterInitStatus::kOperatorRegistrationFailed:
            return "FULLY_CONNECTED operator registration failed";
        case InterpreterInitStatus::kTensorAllocationFailed:
            return "tensor allocation failed; the arena may be too small";
        case InterpreterInitStatus::kTensorCountMismatch:
            return "model input or output tensor count does not match the contract";
        case InterpreterInitStatus::kInputContractMismatch:
            return "input tensor shape, type, size, or quantization does not match";
        case InterpreterInitStatus::kOutputContractMismatch:
            return "output tensor shape, type, size, or quantization does not match";
    }
    return "unknown MicroInterpreter initialization status";
}

tflite::MicroInterpreter* GetInterpreter() {
    if (!g_initialization_succeeded || !g_interpreter.has_value()) {
        return nullptr;
    }
    return &g_interpreter.value();
}

TfLiteTensor* GetInputTensor() {
    return g_initialization_succeeded ? g_input_tensor : nullptr;
}

TfLiteTensor* GetOutputTensor() {
    return g_initialization_succeeded ? g_output_tensor : nullptr;
}

}  // namespace imu::inference
