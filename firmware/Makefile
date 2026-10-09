# 容器把持久化的 TFLM 源码挂载到 /opt/tflm；调用方仍可通过环境变量或命令行覆盖。
TFLM_ROOT ?= /opt/tflm
# 从 Makefile 自身位置推导绝对路径，确保从仓库根目录或 firmware 目录调用 make 时结果
# 一致。构建产物统一放到仓库 build/ 下，避免污染源码目录。
FIRMWARE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(FIRMWARE_DIR)/..)
BUILD_DIR ?= $(PROJECT_ROOT)/build/tflm-host
TARGET := $(BUILD_DIR)/imu_tflm_model_loader
ARENA_PROFILER_TARGET := $(BUILD_DIR)/imu_tflm_arena_profiler
LATENCY_BENCHMARK_TARGET := $(BUILD_DIR)/imu_tflm_latency_benchmark

# FlatBuffers 是 TFLite schema 解析和完整性校验所需的头文件依赖。
FLATBUFFERS_INCLUDE_DIR := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
GEMMLOWP_INCLUDE_DIR := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/gemmlowp

# MicroInterpreter、算子注册和 AllocateTensors() 的实现位于 TFLM 静态库中。固定版本
# TFLM 的 make 系统会把桌面 GCC 产物写到以下目录；若文件不存在，下面的规则会生成。
TFLM_MAKEFILE := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/Makefile
TFLM_LIBRARY := $(TFLM_ROOT)/gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a

# 编译单元分别承载模型字节、FlatBuffer 加载、解释器初始化、单样本推理和桌面入口。
SOURCES := \
	$(PROJECT_ROOT)/firmware/model_data.cc \
	$(PROJECT_ROOT)/firmware/src/inference/model_loader.cc \
	$(PROJECT_ROOT)/firmware/src/inference/tflm_interpreter.cc \
	$(PROJECT_ROOT)/firmware/src/inference/inference_runner.cc \
	$(PROJECT_ROOT)/firmware/src/inference/golden_vectors.cc \
	$(PROJECT_ROOT)/firmware/src/model_loader_main.cc

# 内存分析目标只需要模型加载和 RecordingMicroInterpreter，不链接生产解释器单例。
ARENA_PROFILER_SOURCES := \
	$(PROJECT_ROOT)/firmware/model_data.cc \
	$(PROJECT_ROOT)/firmware/src/inference/model_loader.cc \
	$(PROJECT_ROOT)/firmware/src/arena_profiler_main.cc

LATENCY_BENCHMARK_SOURCES := \
	$(PROJECT_ROOT)/firmware/model_data.cc \
	$(PROJECT_ROOT)/firmware/src/inference/model_loader.cc \
	$(PROJECT_ROOT)/firmware/src/inference/tflm_interpreter.cc \
	$(PROJECT_ROOT)/firmware/src/inference/inference_runner.cc \
	$(PROJECT_ROOT)/firmware/src/inference/golden_vectors.cc \
	$(PROJECT_ROOT)/firmware/src/inference_benchmark_main.cc

# 项目 include 使用普通 -I；第三方目录使用 -isystem，避免上游警告影响本项目构建输出。
CPPFLAGS += \
	-DTF_LITE_STATIC_MEMORY \
	-I$(FIRMWARE_DIR) \
	-I$(FIRMWARE_DIR)/include \
	-isystem $(TFLM_ROOT) \
	-isystem $(FLATBUFFERS_INCLUDE_DIR) \
	-isystem $(GEMMLOWP_INCLUDE_DIR)
# TF_LITE_STATIC_MEMORY 不只是功能开关，也会影响部分 TFLM 公共结构体布局，因此项目
# 编译单元必须与 libtensorflow-microlite.a 使用完全相同的定义以保持 ABI 一致。
# 保持与 CMake 目标相同的 C++ 标准、优化级别和严格警告策略。
CXXFLAGS += -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion

# 静态库必须放在使用其符号的目标文件之后；桌面 TFLM 还会使用标准数学函数。
LDLIBS += $(TFLM_LIBRARY) -lm

.PHONY: all check test profile-arena benchmark-latency clean

all: check $(TARGET)

# 编译前主动验证依赖路径，失败时提示如何初始化环境，而不是等编译器报告模糊错误。
check:
	@test -f "$(TFLM_ROOT)/tensorflow/lite/micro/micro_interpreter.h" || \
		(echo "TFLM source not found at $(TFLM_ROOT)" >&2; exit 1)
	@test -f "$(TFLM_MAKEFILE)" || \
		(echo "TFLM makefile not found at $(TFLM_MAKEFILE)" >&2; exit 1)
	@test -f "$(FLATBUFFERS_INCLUDE_DIR)/flatbuffers/flatbuffers.h" || \
		(echo "FlatBuffers headers are missing; run scripts/setup_tflm_docker.ps1" >&2; exit 1)
	@test -f "$(GEMMLOWP_INCLUDE_DIR)/fixedpoint/fixedpoint.h" || \
		(echo "gemmlowp headers are missing; build the TFLM microlite target first" >&2; exit 1)

# 不把 TFLM 源码复制进项目；需要时直接调用固定源码树自带的构建系统生成 microlite。
# 使用 $(MAKE) 而不是硬编码 make，可继承调用方选择的 GNU Make 程序和并行参数。
$(TFLM_LIBRARY):
	$(MAKE) -C "$(TFLM_ROOT)" -f tensorflow/lite/micro/tools/make/Makefile microlite

# 头文件和 TFLM 库都列为显式依赖；接口、契约或模型变化后会触发重新链接。
$(TARGET): $(SOURCES) \
	$(FIRMWARE_DIR)/model_data.h \
	$(FIRMWARE_DIR)/include/inference/model_loader.h \
	$(FIRMWARE_DIR)/include/inference/model_contract.h \
	$(FIRMWARE_DIR)/include/inference/tflm_interpreter.h \
	$(FIRMWARE_DIR)/include/inference/inference_runner.h \
	$(FIRMWARE_DIR)/include/inference/golden_vectors.h \
	$(TFLM_LIBRARY)
	@mkdir -p "$(BUILD_DIR)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o "$@" $(LDLIBS)

$(ARENA_PROFILER_TARGET): $(ARENA_PROFILER_SOURCES) \
	$(FIRMWARE_DIR)/model_data.h \
	$(FIRMWARE_DIR)/include/inference/model_loader.h \
	$(FIRMWARE_DIR)/include/inference/model_contract.h \
	$(TFLM_LIBRARY)
	@mkdir -p "$(BUILD_DIR)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ARENA_PROFILER_SOURCES) -o "$@" $(LDLIBS)

$(LATENCY_BENCHMARK_TARGET): $(LATENCY_BENCHMARK_SOURCES) \
	$(FIRMWARE_DIR)/model_data.h \
	$(FIRMWARE_DIR)/include/inference/model_loader.h \
	$(FIRMWARE_DIR)/include/inference/model_contract.h \
	$(FIRMWARE_DIR)/include/inference/tflm_interpreter.h \
	$(FIRMWARE_DIR)/include/inference/inference_runner.h \
	$(FIRMWARE_DIR)/include/inference/golden_vectors.h \
	$(TFLM_LIBRARY)
	@mkdir -p "$(BUILD_DIR)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LATENCY_BENCHMARK_SOURCES) -o "$@" $(LDLIBS)

test: all
	# 运行六类 Golden Vector；初始化、Invoke、类别或 INT8 logits 不一致都会失败。
	"$(TARGET)"

profile-arena: check $(ARENA_PROFILER_TARGET)
	# 输出 RecordingMicroAllocator 的真实 head/tail 分配明细，不修改生产 arena。
	"$(ARENA_PROFILER_TARGET)"

benchmark-latency: check $(LATENCY_BENCHMARK_TARGET)
	# 预热后分别测量 Invoke-only 与包含量化/反量化的端到端延迟。
	"$(LATENCY_BENCHMARK_TARGET)"

clean:
	# 只允许删除默认构建目录，防止 BUILD_DIR 被意外设置到项目之外时递归删除错误路径。
	@test "$(BUILD_DIR)" = "$(PROJECT_ROOT)/build/tflm-host" || \
		(echo "Refusing to clean unexpected BUILD_DIR=$(BUILD_DIR)" >&2; exit 1)
	rm -rf "$(BUILD_DIR)"
