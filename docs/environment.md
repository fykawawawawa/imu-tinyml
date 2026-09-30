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
和跨机器运行优先；若后续确实需要 GPU 训练，应按本机 CUDA 版本使用 PyTorch 官方安装器
替换对应 wheel，而不要直接修改其他项目依赖。

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

Python 虚拟环境不负责安装 C++ 编译器、CMake 或 TensorFlow Lite Micro 源码。实现 README
中的桌面端 TFLM `MicroInterpreter` 时，还需要单独配置支持 C++17 的编译器和 CMake；这些
依赖应由后续固件构建配置管理，避免与 Python 包环境混在一起。

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
