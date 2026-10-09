# IMU TinyML HAR

面向资源受限设备的 IMU 人体活动识别项目。项目使用 UCI HAR 的窗口信号提取 39 维
轻量时域特征，训练 `39 -> 32 -> 16 -> 6` 的 PyTorch MLP，将模型转换为全 INT8
TensorFlow Lite，并在桌面端通过 TensorFlow Lite Micro 完成推理验证。

## 实现范围

| 模块 | 本仓库实现 |
| --- | --- |
| 数据与特征 | 读取 UCI HAR 九路窗口信号，按受试者划分数据，提取并标准化 39 维特征 |
| 训练与评估 | 训练 TinyMLP，执行早停，在官方测试集生成分类指标和混淆矩阵 |
| 模型转换 | 完成 PyTorch、Keras、FP32 TFLite 和全 INT8 TFLite 的逐级一致性检查 |
| 桌面 TFLM | 使用静态 tensor arena 完成六类 Golden Vector 校验、内存分析和延迟测试 |

本仓库不包含真实 MCU 固件、传感器驱动、功耗测量或动态频率调节。C++ 推理入口接收已经
标准化的 39 维特征，原始 IMU 信号的滤波和特征提取仍由 Python 完成。

核心流程：

```text
九路窗口信号 (N, 9, 128)
    -> 按受试者划分 train/val，保留官方 test
    -> 提取 39 维特征
    -> 使用 train 参数标准化 train/val/test
    -> TinyMLP 训练与早停
    -> 官方 test 最终评估
    -> 等价 Keras Dense/ReLU
    -> INT8 TFLite
    -> 桌面 TFLM 推理验证
```

## 结果摘要

分类指标来自 UCI HAR 官方测试集（2,947 个窗口、9 名未参与训练的受试者）。INT8 校准
仅使用训练分区；官方测试集不参与校准或模型选择。

| 指标 | FP32 TFLite | 全 INT8 TFLite | INT8 - FP32 |
| --- | ---: | ---: | ---: |
| Accuracy | 87.24% | **87.44%** | +0.20 个百分点 |
| Macro-F1 | 87.03% | **87.24%** | +0.21 个百分点 |
| 模型大小 | 9,580 bytes | **5,600 bytes** | **-41.54%** |

- 模型结构：`39 -> 32 -> 16 -> 6`，共 1,910 个可训练参数；
- FP32 与 INT8 的测试集预测一致率：97.76%（66/2,947 个样本发生变化）；
- Keras 与 FP32 TFLite 的测试集预测一致率：100%；
- 桌面 TFLM 六类 Golden Vector 全部通过，INT8 输出误差不超过 1 LSB；
- TFLM tensor arena 实际分配 1,888 bytes，当前静态预留 2,304 bytes；
- 桌面 host benchmark 的 `Invoke()` P50 为 1.182 us，当前推理 runner P50 为 1.302 us。

INT8 准确率的轻微上升属于该次量化结果，不代表量化通常会提升精度。arena 和延迟均来自
固定 TFLM 版本下的桌面 Docker 环境，不代表真实 MCU 的 RAM 或运行时间。完整测量口径见
[`benchmark/host_latency.md`](benchmark/host_latency.md)。

## 快速开始

项目使用 Python 3.12，依赖版本由 `pyproject.toml` 固定。PowerShell 下执行：

```powershell
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

将 UCI HAR 解压到 `data/raw/UCI HAR Dataset/`，然后运行预处理：

```powershell
python -m training.preprocess
```

训练参数集中保存在 `configs/train.yaml`：

```yaml
epochs: 100
batch_size: 64
learning_rate: 0.001
weight_decay: 0.0001
patience: 15
min_delta: 0.0001
seed: 42
num_workers: 0
device: auto
```

使用默认配置训练：

```powershell
python -m training.train
```

使用最佳检查点在官方测试集上评估：

```powershell
python -m training.evaluate
```

将最佳 PyTorch 检查点转换为等价的 Keras Dense/ReLU 模型，并在官方测试集上验证
logits 与预测类别：

```powershell
python -m training.export_keras
```

默认输出为 `models/exported/best_model.keras` 和
`models/exported/best_model_keras_report.json`；如需覆盖已有输出，添加 `--overwrite`。

将 Keras 桥接模型转换为浮点 TFLite，并验证 TFLite logits：

```powershell
python -m training.export_tflite
```

默认输出为 `models/exported/best_model_float32.tflite` 和对应的转换报告。

使用全部标准化训练输入校准全 INT8 基线模型：

```powershell
python -m training.export_int8_tflite
```

默认输出为 `models/exported/best_model_int8_baseline.tflite` 和
`models/exported/best_model_int8_baseline_report.json`。校准只使用训练分区，候选模型在
验证分区上与 FP32 TFLite 比较。转换命令也支持通过 `--clip-min` 和 `--clip-max` 指定
校准输入的统一裁剪范围；不同设置应使用独立输出路径，避免覆盖基线模型。

在官方测试集上对比 FP32 与 INT8 TFLite，并生成与原评估目录同结构的报告：

```powershell
python -m training.evaluate_int8_tflite
```

默认输出到 `artifacts/int8_evaluation/`，包含 `metrics.json`、INT8 分类报告 CSV、
混淆矩阵 CSV/PNG 和 `evaluation.log`。`metrics.json` 还会记录 FP32/INT8 指标差值、
预测一致率、变化样本数、反量化 logits 误差、输入饱和及输出端点统计。测试集只用于
最终评估，不参与量化校准或裁剪范围选择；如需覆盖现有报告，添加 `--overwrite`。

也可以指定另一份实验配置和输出目录：

```powershell
python -m training.train `
    --config configs/experiment_02.yaml `
    --output-dir models/checkpoints/experiment_02
```

预处理和训练默认拒绝覆盖已有结果；确认需要重新生成时使用 `--overwrite`。运行测试：

```powershell
python -m pytest
python -m ruff check feature_ref models training tests
```

## 数据集与划分

[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones)
包含 30 名受试者的六类日常活动。手机固定在腰部，以 50 Hz 采集三轴加速度和三轴角速度；
数据被切分为 2.56 秒、50% 重叠的窗口，因此每个通道每个窗口包含 128 个采样点。

### 下载

- 数据集主页：[UCI HAR 数据集说明与引用信息](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones)
- 官方压缩包：[human activity recognition using smartphones.zip](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip)
- 外层压缩包 SHA-256：`C00B803081A5C797CD5E4B83700A9810B38D53D9D84E01917E090E1FDBC81031`

官方压缩包中还包含 `UCI HAR Dataset.zip`。将内层压缩包解压到以下位置：

```text
data/raw/UCI HAR Dataset/
```

数据文件不提交到本仓库；使用数据集时请遵循 UCI 页面列出的许可条款并引用原作者。

### 信号

`train/Inertial Signals/` 和 `test/Inertial Signals/` 各包含九路信号：

| 信号 | 通道数 | 单位 | 含义 |
| --- | ---: | --- | --- |
| `body_acc_{x,y,z}` | 3 | g | 去除估计重力后的身体加速度 |
| `body_gyro_{x,y,z}` | 3 | rad/s | 三轴身体角速度 |
| `total_acc_{x,y,z}` | 3 | g | 包含重力分量的总加速度 |

UCI HAR 信号已经过官方滤波和窗口化，并非未经处理的传感器 ADC 数据。真实设备部署时
必须复现等价的滤波、重力分离、采样率和窗口逻辑。

### 活动标签

| 原始标签 | 训练标签 | 活动 |
| ---: | ---: | --- |
| 1 | 0 | `WALKING` |
| 2 | 1 | `WALKING_UPSTAIRS` |
| 3 | 2 | `WALKING_DOWNSTAIRS` |
| 4 | 3 | `SITTING` |
| 5 | 4 | `STANDING` |
| 6 | 5 | `LAYING` |

### 当前划分

官方数据按受试者分为 7,352 个训练窗口和 2,947 个测试窗口。本项目保留官方测试集，
再以默认 `validation_fraction=0.2`、`seed=42` 从官方训练受试者中划分验证集：

| 分区 | 窗口数 | 窗口占比 | 受试者数 | 受试者占比 |
| --- | ---: | ---: | ---: | ---: |
| 训练集 | 5,800 | 56.32% | 17 | 56.67% |
| 验证集 | 1,552 | 15.07% | 4 | 13.33% |
| 测试集 | 2,947 | 28.62% | 9 | 30.00% |
| **合计** | **10,299** | **100.00%** | **30** | **100.00%** |

划分以受试者为单位，同一人不会跨分区。由于每名受试者的窗口数不同，20% 的验证受试者
比例不会产生恰好 20% 的验证窗口；修改随机种子或验证受试者后，窗口比例也会变化。

## 特征与模型

项目不使用 UCI HAR 已生成的 561 维官方特征，而是从九路窗口重新提取 39 维轻量特征：

| 特征组 | 维度 | 内容 |
| --- | ---: | --- |
| 单轴动态统计 | 24 | 6 个动态轴的 mean、std、abs_mean、peak_to_peak |
| 重力均值 | 3 | `total_acc - body_acc` 的 XYZ 均值 |
| 三轴模长统计 | 6 | 加速度和角速度模长的 mean、std、max |
| 轴间相关性 | 6 | 加速度和角速度的 XY、XZ、YZ 相关系数 |
| **合计** | **39** | 顺序由 `feature_ref/feature_schema.py` 固定 |

标准化参数只使用训练子集拟合，再原样应用于验证集和测试集，避免数据泄漏。模型为：

```text
39 -> Linear(32) -> ReLU -> Linear(16) -> ReLU -> Linear(6)
```

模型输出 logits，训练使用交叉熵损失，推理直接取 `argmax`，无需在模型内添加 Softmax。
最佳模型由最低验证损失选择；验证损失连续 `patience` 轮未达到 `min_delta` 改善时早停。

## 生成文件

预处理默认写入 `data/processed/`：

```text
train.npz / val.npz / test.npz    标准化后的特征、标签和受试者编号
standardization.npz               训练集拟合的 mean 和 scale
metadata.json                     特征顺序、划分和预处理配置
```

训练默认写入 `models/checkpoints/`：

```text
best_model.pt    最佳权重、标准化参数、标签映射和完整训练配置
history.json     每轮训练与验证指标
training.log     控制台同步输出的文本日志
```

评估默认写入 `artifacts/evaluation/`，包括 `metrics.json`、分类报告 CSV、混淆矩阵
CSV/PNG 和 `evaluation.log`。INT8 TFLite 对比评估写入 `artifacts/int8_evaluation/`，
文件结构相同，其中分类报告和混淆矩阵对应 INT8 模型。

原始数据、处理结果、检查点、导出模型和评估报告由 `.gitignore` 排除。桌面 TFLM 使用的
INT8 模型字节数组和 Golden Vector 提交到仓库，以便不依赖本地训练产物复现 C++ 验证。

## 目录

```text
feature_ref/    Python 39 维特征参考实现与固定 schema
training/       原始数据读取、预处理和 PyTorch 训练
models/         TinyMLP 定义及本地生成的检查点和导出模型
configs/        训练与转换配置
firmware/       桌面 TFLM 推理、Golden Vector、内存和延迟测试
tests/          Python 单元测试
data/           原始数据和处理后数据
benchmark/      精度、体积、内存和延迟结果
artifacts/      图表与其他实验产物
docs/           环境和设计文档
scripts/        环境、训练、转换和验证入口
```

## 部署边界

项目尚未在真实 MCU 上部署。桌面端 TFLM 结果只代表 host simulation，不能作为
真实板端延迟、RAM、Flash、功耗或动态频率表现。数据来自腰部固定手机和受控实验环境，
模型结果也不能直接代表手腕、鞋部或自由放置设备上的效果。

## License

本项目自行编写的源代码采用 [MIT License](LICENSE)。UCI HAR 数据集、TensorFlow Lite
Micro 及其他第三方组件分别受其各自许可证约束，不因本仓库的 MIT License 而改变。
