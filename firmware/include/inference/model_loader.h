#pragma once

#include <cstddef>

namespace tflite {
// 这里只需要在接口中保存模型指针，因此使用前置声明即可，避免所有包含本头文件的
// 编译单元都被迫引入体积较大的 schema_generated.h。
struct Model;
}  // namespace tflite

namespace imu::inference {

// 将加载过程中的每类失败原因显式编码，调用方可以据此给出有针对性的错误信息，
// 而不必依赖异常或解析文本日志。
enum class ModelLoadStatus {
    // 模型通过长度、文件标识、FlatBuffer 结构和 schema 版本检查。
    kOk,
    // 数据甚至不足以容纳 FlatBuffer 根偏移和四字节的 TFL3 文件标识。
    kModelDataTooSmall,
    // 字节数组存在，但文件标识不是 TensorFlow Lite 使用的 "TFL3"。
    kInvalidFlatbufferIdentifier,
    // 文件标识正确，但表、向量或偏移关系没有通过 FlatBuffers 完整性校验。
    kInvalidFlatbuffer,
    // FlatBuffer 本身有效，但生成模型与当前 TFLM 头文件使用不同 schema 版本。
    kSchemaVersionMismatch,
};

// 无论加载成功还是失败，都返回一份可诊断结果。失败时 model 通常为空；只有 schema
// 不匹配发生在 GetModel() 之后，因此该状态下仍会保留模型视图和实际 schema 版本。
struct ModelLoadResult {
    // 指向 g_model_data 内部的零拷贝 FlatBuffer 视图；不拥有模型内存。
    const tflite::Model* model;
    // 加载和兼容性检查的最终状态。
    ModelLoadStatus status;
    // 编译进程序的原始 .tflite 字节数，用于日志和产物核对。
    std::size_t model_size;
    // 从模型读取的 schema 版本；尚无法安全读取时使用 -1。
    int schema_version;
};

// 将编译进程序的 FlatBuffer 映射为 TFLM 模型视图，并验证它与本次构建使用的 TFLM
// 头文件兼容。该操作不复制模型字节，也不会分配 tensor arena；arena 要到创建
// MicroInterpreter 并调用 AllocateTensors() 时才用于输入、输出和中间张量。
ModelLoadResult LoadEmbeddedModel();

// 把状态枚举转换为稳定、可直接写入日志的英文说明。返回值指向静态字符串，无需释放。
const char* ModelLoadStatusMessage(ModelLoadStatus status);

}  // namespace imu::inference
