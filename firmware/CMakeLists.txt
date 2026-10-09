cmake_minimum_required(VERSION 3.16)

# 这是桌面端 MicroInterpreter 初始化验证目标，不是具体 MCU 的交叉编译工程。它会加载
# 模型、注册算子、分配静态 tensor arena 并检查输入输出契约。
project(imu_tflm_model_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 优先接受 cmake -DTFLM_ROOT=...，未显式传参时再读取同名环境变量。Docker 环境中的
# 默认源码位置是 /opt/tflm，但不在此处写死，以便在其他 Linux 主机上复用。
set(TFLM_ROOT "" CACHE PATH "Path to the TensorFlow Lite Micro source tree")
if(NOT TFLM_ROOT AND DEFINED ENV{TFLM_ROOT})
    set(TFLM_ROOT "$ENV{TFLM_ROOT}")
endif()

# 使用代表性头文件确认 TFLM_ROOT 确实指向已准备好的源码树，尽早给出比编译器
# "header not found" 更清晰的配置错误。
if(NOT EXISTS "${TFLM_ROOT}/tensorflow/lite/micro/micro_interpreter.h")
    message(FATAL_ERROR
        "TFLM_ROOT must point to a prepared TensorFlow Lite Micro source tree. "
        "The Docker environment uses /opt/tflm.")
endif()

# TFLM 的 make 工具会把固定版本的 FlatBuffers 下载到该目录；模型完整性校验所需的
# flatbuffers::Verifier 来自这里。
set(TFLM_FLATBUFFERS_INCLUDE_DIR
    "${TFLM_ROOT}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include")
if(NOT EXISTS "${TFLM_FLATBUFFERS_INCLUDE_DIR}/flatbuffers/flatbuffers.h")
    message(FATAL_ERROR
        "FlatBuffers headers are missing. Run scripts/setup_tflm_docker.ps1 first.")
endif()

# MicroMutableOpResolver 间接包含量化 kernel 头文件，这些头文件使用 gemmlowp 的
# fixedpoint 实现。这里加入 gemmlowp 根目录，而不是 fixedpoint 子目录本身。
set(TFLM_GEMMLOWP_INCLUDE_DIR
    "${TFLM_ROOT}/tensorflow/lite/micro/tools/make/downloads/gemmlowp")
if(NOT EXISTS "${TFLM_GEMMLOWP_INCLUDE_DIR}/fixedpoint/fixedpoint.h")
    message(FATAL_ERROR
        "gemmlowp headers are missing. Build the TFLM microlite target first.")
endif()

# MicroInterpreter 和内置算子实现来自 TFLM make 系统生成的静态库。默认查找 Docker
# 桌面 GCC 产物；其他平台可以通过 -DTFLM_LIBRARY=<文件路径> 显式指定兼容库。
set(TFLM_LIBRARY "" CACHE FILEPATH "Path to libtensorflow-microlite")
if(NOT TFLM_LIBRARY)
    find_library(TFLM_LIBRARY
        NAMES tensorflow-microlite
        PATHS "${TFLM_ROOT}/gen/linux_x86_64_default_gcc/lib"
        NO_DEFAULT_PATH
    )
endif()

if(NOT EXISTS "${TFLM_LIBRARY}")
    message(FATAL_ERROR
        "libtensorflow-microlite was not found. Build it with: "
        "make -C ${TFLM_ROOT} -f tensorflow/lite/micro/tools/make/Makefile microlite")
endif()

# IMPORTED 目标让 CMake 正确处理外部静态库，同时避免把 TFLM 源码误当成本项目源码编译。
add_library(tensorflow_microlite STATIC IMPORTED)
set_target_properties(tensorflow_microlite PROPERTIES
    IMPORTED_LOCATION "${TFLM_LIBRARY}"
)

# model_data.cc 提供只读模型数组，model_loader.cc 负责安全解析，tflm_interpreter.cc
# 负责 resolver、arena 和张量校验，inference_runner.cc 负责量化输入并调用 Invoke，
# golden_vectors.cc 保存由 Python TFLite 生成的六类参考样本和预期 INT8 输出。
add_executable(imu_tflm_model_loader
    model_data.cc
    src/inference/model_loader.cc
    src/inference/tflm_interpreter.cc
    src/inference/inference_runner.cc
    src/inference/golden_vectors.cc
    src/model_loader_main.cc
)

# 单独的内存分析程序使用 RecordingMicroInterpreter 输出实际 arena 分配，不把记录功能
# 带入生产推理目标。其数据源和算子集合与生产目标保持一致。
add_executable(imu_tflm_arena_profiler
    model_data.cc
    src/inference/model_loader.cc
    src/arena_profiler_main.cc
)

# 延迟基准与生产目标复用相同解释器和推理实现；计时入口不进入功能测试目标。
add_executable(imu_tflm_latency_benchmark
    model_data.cc
    src/inference/model_loader.cc
    src/inference/tflm_interpreter.cc
    src/inference/inference_runner.cc
    src/inference/golden_vectors.cc
    src/inference_benchmark_main.cc
)

# TFLM 的 microlite 静态库使用 TF_LITE_STATIC_MEMORY 构建。该宏会影响公共类型布局，
# 因此消费静态库的编译单元也必须使用相同定义，否则读取 TfLiteTensor 时会发生 ABI
# 错位，表现为随机 type/scale/shape，甚至访问无效 dims 指针。
target_compile_definitions(imu_tflm_model_loader PRIVATE
    TF_LITE_STATIC_MEMORY
)
target_compile_definitions(imu_tflm_arena_profiler PRIVATE
    TF_LITE_STATIC_MEMORY
)
target_compile_definitions(imu_tflm_latency_benchmark PRIVATE
    TF_LITE_STATIC_MEMORY
)

# 项目自身头文件使用普通 include 路径，保留本项目代码产生的编译器警告。
target_include_directories(imu_tflm_model_loader PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(imu_tflm_arena_profiler PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(imu_tflm_latency_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# 第三方 TFLM/FlatBuffers 头文件标记为 SYSTEM，避免上游代码的警告淹没本项目警告。
target_include_directories(imu_tflm_model_loader SYSTEM PRIVATE
    "${TFLM_ROOT}"
    "${TFLM_FLATBUFFERS_INCLUDE_DIR}"
    "${TFLM_GEMMLOWP_INCLUDE_DIR}"
)
target_include_directories(imu_tflm_arena_profiler SYSTEM PRIVATE
    "${TFLM_ROOT}"
    "${TFLM_FLATBUFFERS_INCLUDE_DIR}"
    "${TFLM_GEMMLOWP_INCLUDE_DIR}"
)
target_include_directories(imu_tflm_latency_benchmark SYSTEM PRIVATE
    "${TFLM_ROOT}"
    "${TFLM_FLATBUFFERS_INCLUDE_DIR}"
    "${TFLM_GEMMLOWP_INCLUDE_DIR}"
)

# 静态库提供 MicroInterpreter、内存规划器和 FULLY_CONNECTED kernel 的实现。Unix 桌面
# 工具链还需链接 libm；MSVC 的数学函数由标准运行库提供，不存在独立的 m 库。
target_link_libraries(imu_tflm_model_loader PRIVATE tensorflow_microlite)
target_link_libraries(imu_tflm_arena_profiler PRIVATE tensorflow_microlite)
target_link_libraries(imu_tflm_latency_benchmark PRIVATE tensorflow_microlite)
if(UNIX)
    target_link_libraries(imu_tflm_model_loader PRIVATE m)
    target_link_libraries(imu_tflm_arena_profiler PRIVATE m)
    target_link_libraries(imu_tflm_latency_benchmark PRIVATE m)
endif()

# 两类主流编译器都启用严格警告；/permissive- 让 MSVC 更接近标准 C++ 行为。
if(MSVC)
    target_compile_options(imu_tflm_model_loader PRIVATE /W4 /permissive-)
    target_compile_options(imu_tflm_arena_profiler PRIVATE /W4 /permissive-)
    target_compile_options(imu_tflm_latency_benchmark PRIVATE /W4 /permissive-)
else()
    target_compile_options(imu_tflm_model_loader PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion
    )
    target_compile_options(imu_tflm_arena_profiler PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion
    )
    target_compile_options(imu_tflm_latency_benchmark PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion
    )
endif()

# 注册六类 Golden Vector 一致性测试：类别必须相同，INT8 logits 最大允许相差 1 LSB。
enable_testing()
add_test(NAME tflm_golden_vectors COMMAND imu_tflm_model_loader)
