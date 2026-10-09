#pragma once
#include <cstddef>
#include <cstdint>

// TFLite FlatBuffer 的只读字节数组及其长度。数组定义在 model_data.cc 中，避免每个
// 包含本头文件的编译单元都复制一份模型数据。
extern const std::uint8_t g_model_data[];
extern const std::size_t g_model_data_len;
