# IMU TinyML HAR

面向资源受限设备的 IMU 人体活动识别项目。项目使用 UCI HAR 的窗口信号提取 39 维
轻量时域特征，训练 `39 -> 32 -> 16 -> 6` 的 PyTorch MLP，并计划转换为全 INT8
TensorFlow Lite 模型，在桌面端通过 TensorFlow Lite Micro 验证推理流程。

## 当前状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 数据读取 | 已完成 | 读取九路信号、标签和受试者编号，校验窗口与样本对齐 |
| 特征工程 | 已完成 | 提取固定顺序的 39 维特征，并保存标准化参数 |
| 模型训练 | 已完成 | 训练、验证、早停、指标记录和最佳检查点保存 |
| 测试集评估 | 已完成 | 总体/逐类指标、分类报告和混淆矩阵 |
| Keras 转换 | 已完成 | PyTorch 权重迁移、落盘重载及全测试集 logits 对齐 |
| 浮点 TFLite | 已完成 | Keras 到 FP32 FlatBuffer 转换及全测试集 logits 对齐 |
| INT8 基线 | 已完成 | 使用全部原始标准化训练输入校准全 INT8 模型 |
| 裁剪校准 | 待比较 | 已预留统一上下界策略，后续比较 `±8`、`±6` 等方案 |
| TFLM 推理 | 已完成（桌面） | MicroInterpreter、静态 arena 和六类 Golden Vector 的 INT8 输出已验证 |

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
    -> TFLM 推理
```

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
验证分区上与 FP32 TFLite 比较。转换入口已预留裁剪策略；后续实验应为不同策略指定独立
输出路径，例如同时传入 `--clip-min -8 --clip-max 8`。

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

数据、处理结果、检查点和其他生成物均由 `.gitignore` 排除。

## 目录

```text
feature_ref/    Python 39 维特征参考实现与固定 schema
training/       原始数据读取、预处理和 PyTorch 训练
models/         TinyMLP 定义、检查点及后续导出模型
configs/        训练与转换配置
firmware/       C++ 特征和 TFLM 推理代码
tests/          Python 单元测试与后续 golden vectors
data/           原始数据和处理后数据
benchmark/      精度、体积、内存和延迟结果
artifacts/      图表与其他实验产物
docs/           环境和设计文档
scripts/        环境、训练、转换和验证入口
```

## 部署边界

当前项目尚未在真实 MCU 上部署。后续桌面端 TFLM 结果只代表 host simulation，不能作为
真实板端延迟、RAM、Flash、功耗或动态频率表现。数据来自腰部固定手机和受控实验环境，
模型结果也不能直接代表手腕、鞋部或自由放置设备上的效果。
