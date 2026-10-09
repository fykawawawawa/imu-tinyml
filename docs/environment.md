# Python 虚拟环境

## 版本选择

项目固定使用 64 位 Python 3.12。该版本同时满足 Windows 下 PyTorch 训练和
TensorFlow CPU/TFLite 转换的要求。依赖版本统一定义在根目录 `pyproject.toml` 中：

- NumPy：特征提取和数组处理；
- PyTorch：轻量 MLP 训练与 FP32 推理；
- scikit-learn：特征标准化、分类指标和混淆矩阵；
- Matplotlib：训练曲线和评估图表；
- TensorFlow：Keras 模型承接及全 INT8 TFLite 转换；
- pytest、Ruff：测试和代码检查。

原生 Windows TensorFlow 2.11 之后仅支持 CPU。这个项目只用 TensorFlow 执行模型转换
和桌面端校验，因此不依赖 TensorFlow GPU。默认安装的 PyTorch 也是 CPU 版本，以可复现
和跨机器运行优先。需要 GPU 训练时，按本机 CUDA 版本使用 PyTorch 官方安装器替换对应
wheel，其余依赖保持不变。

## 自动配置

在 PowerShell 中进入仓库根目录并运行：

```powershell
.\scripts\setup_env.ps1
```

脚本会完成以下操作：

1. 在项目根目录创建 `.venv`；
2. 更新虚拟环境中的 pip、setuptools 和 wheel；
3. 以 editable 模式安装项目及完整依赖；
4. 运行 Python 测试。

只进行特征开发、不安装 PyTorch 和 TensorFlow 时，可以使用轻量配置：

```powershell
.\scripts\setup_env.ps1 -Profile minimal
```

## 手动配置

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[all]"
python -m pytest
```

若 PowerShell 执行策略禁止激活脚本，不必修改系统策略，可以直接使用：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 常用检查

```powershell
python -m pytest
python -m ruff check feature_ref tests training scripts
python -c "import numpy, torch, sklearn, tensorflow; print('environment ok')"
```

## 非 Python 依赖

Python 虚拟环境不安装 C++ 编译器、CMake 或 TensorFlow Lite Micro 源码。桌面端 TFLM
验证使用 `compose.yaml` 固定工具链镜像，并通过 `firmware/Makefile` 构建；CMake 文件用于
已有本地 TFLM 源码和静态库的环境。

## Docker Desktop + WSL2 的 TFLM 环境

Windows 主机使用 Docker Desktop 的 WSL2 Linux 容器后端，不需要再安装完整的 Ubuntu
虚拟机。仓库根目录的 `compose.yaml` 使用 TFLM 官方 CI 镜像
`ghcr.io/tflm-bot/tflm-ci:0.6.7`，其中包含 Clang、Bazel 和 QEMU。TFLM 源码保存在 Docker
命名卷 `imu-process-tflm-source` 中，与数据集、Python 虚拟环境和训练产物隔离。

首次配置并运行官方 `hello_world` 冒烟测试：

```powershell
.\scripts\setup_tflm_docker.ps1
```

只初始化环境、不编译冒烟测试：

```powershell
.\scripts\setup_tflm_docker.ps1 -SkipSmokeTest
```

进入容器终端：

```powershell
docker compose run --rm tflm bash
```

容器内的项目路径为 `/workspace`，TFLM 源码路径为 `/opt/tflm`。源码固定到提交
`cd32a519356a05615e58001637ee25f73b009c86`，以保证构建结果可复现。

### 验证内嵌 INT8 模型与推理一致性

模型重新导出后，先同步生成 C++ 字节数组：

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_model_data.py
.\.venv\Scripts\python.exe .\scripts\generate_tflm_golden_vectors.py
```

然后在已经启动的 TFLM 容器中编译并运行桌面加载器：

```powershell
docker compose exec tflm bash -lc "cd /workspace && make -C firmware test"
```

第一条脚本把完整 INT8 FlatBuffer 转换为 C++ 数组；第二条脚本在官方测试集中为六个活动
类别各选择一个被 Python TFLite 正确分类的确定性样本，并保存标准化特征和预期 INT8
logits。每次重新导出模型后都必须重新运行这两条脚本。

桌面程序检查 FlatBuffer、schema、输入输出契约和静态 tensor arena，随后使用
`MicroMutableOpResolver<1>` 注册 `FULLY_CONNECTED`、调用 `Invoke()`，并逐项比较 TFLM
与 Python TFLite 的输出。预测类别必须完全相同，六个 INT8 logits 最大允许相差 1 LSB。
当前基线模型的预期摘要为：

```text
MicroInterpreter initialized successfully.
Tensor arena reserved: 2304 bytes
Input tensor: int8 [1, 39], bytes=39, scale=0.147421, zero_point=31
Output tensor: int8 [1, 6], bytes=6, scale=0.23945, zero_point=19
Golden Vector validation passed: 6/6 samples.
```

Golden Vector 验证证明桌面 TFLM 与 Python TFLite 的量化输入、算子执行、输出读取和
argmax 链路一致，但不替代完整测试集准确率评估，也不代表真实 MCU 的延迟、RAM、Flash
或功耗。安装了 CMake 的环境可以使用 `firmware/CMakeLists.txt`，并通过
`-DTFLM_ROOT=<TFLM 源码路径>` 指定源码目录；固定的 TFLM CI 镜像本身不包含 CMake，
因此容器内默认使用 Makefile。

### 测量 tensor arena

模型或 TFLM 版本变化后，运行独立的 `RecordingMicroInterpreter` 分析目标：

```powershell
docker compose exec tflm bash -lc "cd /workspace && make -C firmware profile-arena"
```

当前固定模型和 TFLM 版本的实测分配为 1,888 bytes，其中 head 80 bytes、tail 1,808
bytes。生产配置使用 2,304 bytes，保留 416 bytes（约 22%）余量并保持16字节对齐。修改
arena 后必须重新运行 `make -C firmware test`；如果模型结构、算子集合或 TFLM 提交变化，
不能继续沿用旧测量值。

### 测量桌面 TFLM 推理延迟

延迟目标先预热 1,000 次，再对六类 Golden Vector 循环测量 50,000 次：

```powershell
docker compose exec tflm bash -lc "cd /workspace && make -C firmware benchmark-latency"
```

输出分别包含纯 `MicroInterpreter::Invoke()` 和当前特征级端到端 `RunInference()` 的
mean、P50、P95、P99 及最大/最小值。端到端计时包含输入检查、INT8 量化、推理、输出
反量化和 `argmax`，不包含原始 IMU 特征提取与标准化。固定单核的五轮
实测结果及完整环境记录见 `benchmark/host_latency.md`。
