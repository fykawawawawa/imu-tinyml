#include "inference/model_loader.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace imu::inference {
namespace {

// 标准 TFLite FlatBuffer 的开头至少需要：
//   1. 四字节根表偏移；
//   2. 四字节文件标识 "TFL3"。
// 在调用任何 FlatBuffers/TFLite 解析函数前先检查最小长度，可避免对明显截断的数据执行
// 越界读取。这个检查只排除“不可能有效”的输入，完整结构仍由 Verifier 负责验证。
constexpr std::size_t kMinimumFlatbufferSize = 8;

}  // namespace

ModelLoadResult LoadEmbeddedModel() {
    // 第一层：检查嵌入数组是否至少包含 FlatBuffer 的固定头部。此时还不能安全读取
    // schema，因此 schema_version 使用 -1 表示“未知”。
    if (g_model_data_len < kMinimumFlatbufferSize) {
        return {nullptr, ModelLoadStatus::kModelDataTooSmall, g_model_data_len, -1};
    }

    // 第二层：快速核对字节 4～7 是否为 TFLite 文件标识 "TFL3"。它能识别选错模型、
    // 生成脚本输入错误或数组内容被替换，但不能证明其余 FlatBuffer 结构完整。
    if (!tflite::ModelBufferHasIdentifier(g_model_data)) {
        return {
            nullptr,
            ModelLoadStatus::kInvalidFlatbufferIdentifier,
            g_model_data_len,
            -1,
        };
    }

    // 第三层：使用数组的真实长度验证整个 FlatBuffer。Verifier 会检查所有表偏移、
    // 向量长度及嵌套对象边界，防止 GetModel() 返回一个随后访问会越界的模型视图。
    flatbuffers::Verifier verifier(g_model_data, g_model_data_len);
    if (!tflite::VerifyModelBuffer(verifier)) {
        return {
            nullptr,
            ModelLoadStatus::kInvalidFlatbuffer,
            g_model_data_len,
            -1,
        };
    }

    // GetModel() 只在 g_model_data 上建立只读 FlatBuffer 视图，不分配内存，也不会把
    // 5,600 字节模型复制进 tensor arena。g_model_data 是静态数组，生命周期覆盖整个
    // 程序，因此返回的 model 指针在程序运行期间始终有效。
    const tflite::Model* model = tflite::GetModel(g_model_data);

    // 第四层：模型文件和当前 TFLM 代码必须使用同一 schema 版本。即使 FlatBuffer
    // 结构完全合法，版本不一致也可能让运行时误解字段含义，因此不能继续创建解释器。
    const int schema_version = model->version();
    if (schema_version != TFLITE_SCHEMA_VERSION) {
        return {
            model,
            ModelLoadStatus::kSchemaVersionMismatch,
            g_model_data_len,
            schema_version,
        };
    }

    // 至此仅代表“模型可安全读取且 schema 兼容”。算子注册、tensor arena 分配以及
    // 输入输出契约检查属于后续 MicroInterpreter 初始化阶段。
    return {
        model,
        ModelLoadStatus::kOk,
        g_model_data_len,
        schema_version,
    };
}

const char* ModelLoadStatusMessage(ModelLoadStatus status) {
    // switch 覆盖所有已知枚举值；末尾仍提供兜底字符串，以防收到由损坏数据或未来版本
    // 新增枚举产生的未知值。
    switch (status) {
        case ModelLoadStatus::kOk:
            return "model loaded successfully";
        case ModelLoadStatus::kModelDataTooSmall:
            return "model data is too small to contain a TFLite FlatBuffer";
        case ModelLoadStatus::kInvalidFlatbufferIdentifier:
            return "model data does not have the TFL3 FlatBuffer identifier";
        case ModelLoadStatus::kInvalidFlatbuffer:
            return "model data is not a valid TFLite FlatBuffer";
        case ModelLoadStatus::kSchemaVersionMismatch:
            return "model schema version is incompatible with this TFLM build";
    }
    return "unknown model load status";
}

}  // namespace imu::inference
