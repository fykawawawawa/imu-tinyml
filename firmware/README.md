# Firmware：桌面端 TFLite Micro 推理

本目录负责把全 INT8 TFLite 模型嵌入 C++，并使用 TensorFlow Lite Micro（TFLM）完成
模型校验、解释器初始化、单样本推理、Golden Vector 一致性测试、静态内存分析和 host
延迟基准。

当前实现是桌面端 host simulation，不包含 MCU 启动代码、IMU 驱动、滤波器或 39 维特征
提取。`RunInference()` 的输入必须是已经使用训练集参数标准化的 39 维浮点特征。

## 目录结构

```text
firmware/
├── README.md                         本文档
├── Makefile                          Docker/GNU Make 构建与运行入口
├── CMakeLists.txt                    CMake 构建定义和 CTest 注册
├── model_data.h                      内嵌模型数组和长度的声明
├── model_data.cc                     5,600 字节 INT8 TFLite FlatBuffer
├── include/
│   ├── inference/
│   │   ├── model_contract.h          模型输入、输出、量化参数和 arena 契约
│   │   ├── model_loader.h            FlatBuffer 加载接口和错误状态
│   │   ├── tflm_interpreter.h        MicroInterpreter 初始化和访问接口
│   │   ├── inference_runner.h        单样本推理接口和结果结构
│   │   └── golden_vectors.h          六类 Golden Vector 数据声明
│   ├── features/.gitkeep             预留：C++ 特征提取接口
│   ├── filters/.gitkeep              预留：滤波和重力分离接口
│   ├── imu/.gitkeep                  预留：IMU 采样、窗口和缓冲接口
│   └── platform/.gitkeep             预留：MCU/操作系统平台适配接口
├── src/
│   ├── model_loader_main.cc          功能测试和 Golden Vector 验证入口
│   ├── arena_profiler_main.cc        tensor arena 分配分析入口
│   ├── inference_benchmark_main.cc   初始化、纯推理和端到端延迟基准入口
│   ├── inference/
│   │   ├── model_loader.cc           模型完整性和 schema 兼容性校验
│   │   ├── tflm_interpreter.cc       resolver、arena、解释器和张量契约
│   │   ├── inference_runner.cc       输入量化、Invoke、输出处理和分类
│   │   └── golden_vectors.cc         六类 Golden Vector 数据定义
│   ├── features/.gitkeep             预留：C++ 特征提取实现
│   ├── filters/.gitkeep              预留：滤波实现
│   ├── imu/.gitkeep                  预留：传感器与滑动窗口实现
│   └── platform/.gitkeep             预留：板级实现
└── third_party/.gitkeep              预留；TFLM 当前由外部源码树提供
```

## 分层与文件职责

### 1. 模型数据与契约

| 文件 | 作用 | 关键内容 |
| --- | --- | --- |
| `model_data.h/.cc` | 把 `.tflite` 模型编译进程序 | `g_model_data`、`g_model_data_len`；数组按 16 字节对齐 |
| `model_contract.h` | 固定 C++ 运行时必须满足的模型契约 | 输入 `[1,39]`、输出 `[1,6]`、INT8 scale/zero point、2,304-byte arena |
| `golden_vectors.h/.cc` | 保存 Python TFLite 产生的参考输入和输出 | 六个活动类别各一个已标准化样本及预期 INT8 logits |

当前量化契约：

| 张量 | Shape | 类型 | Scale | Zero point |
| --- | --- | --- | ---: | ---: |
| 输入 | `[1, 39]` | INT8 | `0.1474212110042572` | `31` |
| 输出 | `[1, 6]` | INT8 | `0.23944978415966034` | `19` |

`model_data.h/.cc` 由 `scripts/generate_model_data.py` 从
`models/exported/best_model_int8_baseline.tflite` 生成；`golden_vectors.h/.cc` 由
`scripts/generate_tflm_golden_vectors.py` 根据同一模型和官方测试集生成。

### 2. 模型加载层

`model_loader.h/.cc` 提供 `LoadEmbeddedModel()`。它按顺序执行：

1. 检查模型数组至少包含 FlatBuffer 固定头部；
2. 检查 `TFL3` 文件标识；
3. 使用 `flatbuffers::Verifier` 校验表、向量和偏移边界；
4. 使用 `tflite::GetModel()` 建立零拷贝模型视图；
5. 对比模型 schema 与当前 TFLM 的 `TFLITE_SCHEMA_VERSION`。

这一层不复制模型、不注册算子，也不分配 tensor arena。调用方通过 `ModelLoadStatus` 区分
数据过短、标识错误、FlatBuffer 损坏和 schema 不兼容。

### 3. 解释器初始化层

`tflm_interpreter.h/.cc` 提供以下公共接口：

| 接口 | 作用 |
| --- | --- |
| `InitializeInterpreter()` | 加载模型、注册算子、创建解释器、分配张量并检查契约 |
| `GetInterpreter()` | 返回进程内唯一的 `MicroInterpreter` |
| `GetInputTensor()` | 返回通过契约校验的输入张量 |
| `GetOutputTensor()` | 返回通过契约校验的输出张量 |
| `InterpreterInitStatusMessage()` | 将初始化状态转换为日志文本 |

模型有三个全连接节点，但只使用一种算子，因此使用
`MicroMutableOpResolver<1>` 并仅调用 `AddFullyConnected()`。ReLU 已融合在
`FULLY_CONNECTED` 节点中，不需要单独注册。

解释器、resolver、输入/输出张量指针和 `alignas(16)` tensor arena 都由该模块静态持有。
arena 容量为 2,304 bytes；`RecordingMicroInterpreter` 实测使用 1,888 bytes，剩余 416
bytes 作为对齐及平台差异余量。

初始化只尝试一次，后续调用返回第一次的结果。初始化成功前，三个对象访问接口均返回
空指针；如果第一次初始化失败，需要修复模型或配置后重启进程。

### 4. 推理执行层

`inference_runner.h/.cc` 提供：

```cpp
InferenceStatus RunInference(
    const float* normalized_features,
    std::size_t feature_count,
    InferenceResult* result);
```

调用前必须成功执行 `InitializeInterpreter()`。推理过程如下：

```text
39 维已标准化 float 特征
    │ 检查指针、长度以及 NaN/Inf
    ▼
q = clamp(round_to_even(real / input_scale) + input_zero_point, -128, 127)
    │ 写入 input->data.int8
    ▼
MicroInterpreter::Invoke()
    │ 读取 6 个 INT8 logits
    ├─ 直接在 INT8 域 argmax 得到 predicted_class
    └─ real = (q - output_zero_point) × output_scale
       得到 dequantized_logits
```

`InferenceResult` 同时返回预测类别、六个 INT8 logits 和六个反量化 logits。解释器和张量
是进程级共享状态，当前接口不支持并发或可重入调用。

## 可执行程序与调用关系

### 功能测试：`imu_tflm_model_loader`

```text
model_loader_main.cc
  └─ InitializeInterpreter()
       ├─ LoadEmbeddedModel()
       │    └─ g_model_data / g_model_data_len
       ├─ AddFullyConnected()
       ├─ MicroInterpreter(..., static arena)
       ├─ AllocateTensors()
       └─ 校验 input/output 数量、shape、dtype、bytes、scale、zero point
  └─ 遍历 kGoldenVectors
       └─ RunInference()
            ├─ GetInterpreter()/GetInputTensor()/GetOutputTensor()
            ├─ float → INT8
            ├─ Invoke()
            └─ INT8 logits → 反量化 logits + argmax
```

程序对六类参考样本执行真实推理，要求预测类别完全一致，并允许每个 INT8 logit 与 Python
TFLite 参考值最多相差 1 LSB。任一初始化、推理或一致性检查失败都会返回非零退出码。

### 内存分析：`imu_tflm_arena_profiler`

```text
arena_profiler_main.cc
  └─ LoadEmbeddedModel()
  └─ AddFullyConnected()
  └─ RecordingMicroInterpreter
       ├─ AllocateTensors()
       └─ GetMicroAllocator().PrintAllocations()
```

该程序使用独立 arena 和 `RecordingMicroInterpreter`，不会复用生产解释器。模型、TFLM
版本或算子集合变化后，应重新运行并据此更新 `kTensorArenaSize`。

### 延迟基准：`imu_tflm_latency_benchmark`

```text
inference_benchmark_main.cc
  ├─ InitializeInterpreter()                 初始化延迟
  ├─ 预热 1,000 次
  ├─ interpreter->Invoke() × 50,000          纯推理延迟
  ├─ RunInference() × 50,000                 特征级端到端延迟
  └─ 输出 min/mean/P50/P95/P99/max
```

纯推理计时不包含输入量化；特征级端到端计时包含输入检查、量化、`Invoke()`、反量化和
`argmax`，但不包含原始 IMU 采样、滤波、特征提取或标准化。详细测量环境和结果保存在
`benchmark/host_latency.md`。

## 构建依赖

- C++17 编译器；
- 固定版本的 TFLM 源码树；
- TFLM make 系统下载的 FlatBuffers 和 gemmlowp 头文件；
- `libtensorflow-microlite.a`；
- Unix 桌面目标所需的数学库 `libm`。

所有目标都定义 `TF_LITE_STATIC_MEMORY`。该宏会影响 TFLM 公共类型布局，项目编译单元
必须与 `libtensorflow-microlite.a` 使用一致定义，否则可能错误读取 `TfLiteTensor`。

## 使用 GNU Make

先按仓库的 [环境说明](../docs/environment.md)准备 Docker 和 `/opt/tflm`。在仓库根目录
运行：

```powershell
# 编译并运行 Golden Vector 功能测试
docker compose exec tflm bash -lc "cd /workspace && make -C firmware test"

# 输出 tensor arena 分配明细
docker compose exec tflm bash -lc "cd /workspace && make -C firmware profile-arena"

# 测量纯 Invoke 和特征级端到端延迟
docker compose exec tflm bash -lc "cd /workspace && make -C firmware benchmark-latency"
```

常用 Make 目标：

| 目标 | 作用 | 产物或行为 |
| --- | --- | --- |
| `all` | 检查依赖并构建功能测试 | `build/tflm-host/imu_tflm_model_loader` |
| `test` | 构建并运行六类 Golden Vector | 失败时返回非零退出码 |
| `profile-arena` | 构建并运行内存分析 | `imu_tflm_arena_profiler` |
| `benchmark-latency` | 构建并运行延迟基准 | `imu_tflm_latency_benchmark` |
| `clean` | 删除默认的 `build/tflm-host` | 拒绝清理被改写到其他位置的 `BUILD_DIR` |

若 `libtensorflow-microlite.a` 不存在，Makefile 会调用固定 TFLM 源码树自带的 make 系统
生成。默认 `TFLM_ROOT=/opt/tflm`，可在命令行覆盖。

## 使用 CMake

`CMakeLists.txt` 定义与 Makefile 相同的三个可执行程序，并把功能测试注册为
`tflm_golden_vectors`。配置时必须指定已准备好的 TFLM 源码和兼容静态库：

```bash
cmake -S firmware -B build/tflm-cmake \
  -DTFLM_ROOT=/path/to/tflite-micro \
  -DTFLM_LIBRARY=/path/to/libtensorflow-microlite.a
cmake --build build/tflm-cmake
ctest --test-dir build/tflm-cmake --output-on-failure
```

固定的 TFLM Docker CI 镜像当前不包含 CMake，因此容器内默认使用 GNU Make。

## 更新模型后的同步流程

重新导出 `best_model_int8_baseline.tflite` 后，应按以下顺序更新：

1. 运行 `scripts/generate_model_data.py`，重新生成 `model_data.h/.cc`；
2. 检查并更新 `model_contract.h` 的 shape、scale 和 zero point；
3. 运行 `scripts/generate_tflm_golden_vectors.py`，重新生成 Golden Vector；
4. 运行 `make -C firmware profile-arena`，重新测量并设置 arena；
5. 运行 `make -C firmware test`，验证 TFLM 与 Python TFLite 一致；
6. 运行 `make -C firmware benchmark-latency`，刷新 host 延迟结果。

模型数组、Golden Vector 和 `model_contract.h` 当前不是单一脚本原子生成；提交前必须确认
三者来自同一 `.tflite` 文件，避免模型与硬编码契约失配。

## 当前边界与后续目录

- `features/`、`filters/`、`imu/` 和 `platform/` 目前只有占位文件；
- 当前“端到端”从已标准化 39 维特征开始，不是从六轴 IMU 原始采样开始；
- 当前没有 MCU 交叉编译、启动文件、链接脚本、传感器驱动或动态频率调节实现；
- host 上测得的 arena 和延迟不能直接代表真实 MCU 的 RAM、Flash、延迟或功耗；
- 若后续接入 MCU，应保持 `model_loader`、`tflm_interpreter` 和 `inference_runner` 接口，
  在预留目录中补充传感器、滤波、特征和平台适配层。
