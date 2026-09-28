# IMU TinyML HAR

面向资源受限设备的无板 IMU 行为识别项目，计划在四天内完成从时频域特征、PyTorch 训练、INT8 量化到桌面端 TensorFlow Lite Micro 推理的完整软件闭环。

## 当前状态

项目骨架、UCI HAR 原始数据、39 维时域特征参考实现和 Python 环境配置已就绪；尚未执行
完整数据预处理、模型训练、量化或基准测试。

## 轻量 MLP 训练任务

训练目标是使用 UCI HAR 的九路惯性信号完成六类人体活动识别。每个样本包含九个通道、
每通道 128 个采样点；先通过 `feature_ref/extractor.py` 提取固定顺序的 39 维时域特征，
再输入 `models/mlp.py` 中的 `39 -> 32 -> 16 -> 6` 轻量 MLP。模型输出 logits，训练时
直接使用交叉熵损失；原始 `1～6` 标签需转换为 `0～5`。

训练流程应遵循以下约束：

- 使用 `Inertial Signals/` 中的窗口信号重新提取 39 维特征，不直接使用 561 维官方特征；
- 保留官方测试集用于最终评估，并从官方训练集按受试者划分训练集和验证集；
- 特征标准化参数只能由训练子集拟合，再应用于验证集和测试集；
- 检查点需同时保存模型权重、标准化参数、特征顺序、标签映射和训练配置；
- 最终报告准确率、各类别 precision/recall/F1 和混淆矩阵。

当前已有模型定义、单窗口特征提取和接口测试。完成训练流水线还需补充：

| 建议文件 | 职责 |
| --- | --- |
| `training/data.py` | 批量读取九路信号、标签和受试者编号，生成 Dataset/DataLoader |
| `training/preprocess.py` | 提取并标准化 39 维特征，保存处理结果和标准化参数 |
| `training/train.py` | 训练与验证、早停、日志记录和最佳检查点保存 |
| `training/evaluate.py` | 加载最佳模型，在官方测试集上生成分类指标和混淆矩阵 |
| `configs/train.yaml` | 保存数据路径、随机种子和训练超参数 |
| `scripts/train.ps1` | 提供预处理、训练和评估的统一执行入口 |

其中 `data.py`、`train.py` 和 `evaluate.py` 是最小可运行训练流程所需的核心文件。

## Python 环境

项目使用 Python 3.12，并通过根目录 `pyproject.toml` 固定 Python 依赖。PowerShell 下运行：

```powershell
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

默认安装 README 完整任务链需要的 NumPy、PyTorch、scikit-learn、Matplotlib、TensorFlow、
pytest 和 Ruff，并在安装后自动执行测试。只进行特征开发时可运行
`.\scripts\setup_env.ps1 -Profile minimal`。详细说明见 `docs/environment.md`。

## UCI HAR 数据集

本项目使用 [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones) 数据集。数据由 30 名年龄为 19～48 岁的志愿者采集；实验过程中，志愿者将 Samsung Galaxy S II 智能手机佩戴在腰部，并完成六种日常活动。手机内置加速度计和陀螺仪以 50 Hz 采集三轴线加速度和三轴角速度，活动标签由实验录像人工标注。

数据集在本地保存于 `data/raw/UCI HAR Dataset/`，原始数据文件由根目录 `.gitignore` 排除，不提交到 Git。

### 基本属性

| 属性 | 内容 |
| --- | --- |
| 任务类型 | 多变量时序数据的六分类人体活动识别 |
| 受试者 | 30 人，年龄 19～48 岁 |
| 采集设备 | 腰部佩戴的 Samsung Galaxy S II |
| 传感器 | 三轴加速度计、三轴陀螺仪 |
| 采样频率 | 50 Hz |
| 窗口长度 | 2.56 秒，即每轴 128 个采样点 |
| 窗口重叠 | 50% |
| 样本总数 | 10,299 个窗口 |
| 训练集 | 7,352 个窗口，来自 21 名受试者 |
| 测试集 | 2,947 个窗口，来自另外 9 名受试者 |
| 官方特征 | 每个窗口 561 维时域与频域特征，归一化至 `[-1, 1]` |

官方划分按受试者进行：约 70% 的受试者构成训练集，其余约 30% 构成测试集。因此同一受试者不会同时出现在训练集和测试集中，可以减少人体个体特征泄漏造成的虚高结果。本项目保留这一官方划分，不对窗口进行随机重划分。

### 活动标签

| 标签 | 官方名称 | 中文含义 |
| ---: | --- | --- |
| 1 | `WALKING` | 步行 |
| 2 | `WALKING_UPSTAIRS` | 上楼 |
| 3 | `WALKING_DOWNSTAIRS` | 下楼 |
| 4 | `SITTING` | 坐姿 |
| 5 | `STANDING` | 站立 |
| 6 | `LAYING` | 躺卧 |

### 原始惯性信号

`train/Inertial Signals/` 与 `test/Inertial Signals/` 各包含 9 个信号文件。
Inertial Signals中各个通道中的一行为128个采样点。每行对应一个时序窗口，每行有 128 个采样值。

| 信号组 | 轴数 | 单位 | 含义 |
| --- | ---: | --- | --- |
| `total_acc_{x,y,z}` | 3 | 标准重力加速度 `g` | 加速度计测得的总加速度，包含身体运动与重力分量 |
| `body_acc_{x,y,z}` | 3 | 标准重力加速度 `g` | 从总加速度中去除估计重力分量后的身体加速度 |
| `body_gyro_{x,y,z}` | 3 | rad/s | 陀螺仪测得的三轴角速度 |

其中 `1 g = 9.80665 m/s²`。数据集不包含磁力计数据，因此不能直接用于完整的九轴姿态融合；本项目主要使用上述六轴身体加速度与角速度完成活动识别。

### 官方预处理

官方数据并非完全未经处理的传感器 ADC 输出，其处理过程包括：

1. 对 50 Hz 三轴加速度和角速度信号进行中值滤波，并使用截止频率为 20 Hz 的三阶 Butterworth 低通滤波器去噪。
2. 使用截止频率为 0.3 Hz 的 Butterworth 低通滤波器估计重力分量，并由总加速度得到身体加速度。
3. 采用 128 点、50% 重叠的固定滑动窗口进行切分。
4. 对身体加速度和角速度求导得到 jerk 信号，并计算三轴信号的欧氏模长。
5. 对部分信号执行 FFT，形成频域信号和频域统计量。

这意味着本项目进行自定义特征工程时，将从 `Inertial Signals/` 中的窗口信号出发，但需要在结果说明中注明这些信号已经经过官方滤波与窗口化。

### 561 维官方特征

官方的 `X_train.txt` 和 `X_test.txt` 已为每个窗口提供 561 维特征。特征名前缀 `t` 表示时域，`f` 表示经过 FFT 得到的频域。主要中间信号包括：

- 身体加速度、重力加速度及身体加速度 jerk；
- 身体角速度及角速度 jerk；
- 上述三轴信号的欧氏模长；
- 身体加速度、加速度 jerk、角速度及其模长的频域表示。

#### 窗口内 128 个采样点与窗口维度 561 个维度的对应关系

`128` 和 `561` 描述的是两个不同概念：

- `128` 是一个 2.56 秒窗口内**每个轴的时序采样点数**，因为 `50 Hz × 2.56 s = 128`。一个六轴窗口最初可以看成 `6 × 128` 个传感器读数。
- `561` 是对整个窗口进行信号处理和统计汇总后得到的**特征标量数量**，不是把 128 个采样点直接扩展或补齐到 561。

官方流水线会先从六轴测量生成更多中间信号，例如身体加速度、重力加速度、jerk、三轴模长以及 FFT 频谱；然后对不同信号、不同轴分别计算均值、标准差、能量、熵、相关系数和频带能量等统计量。每一次“信号 × 轴 × 统计量”的组合产生一个标量特征，所有标量拼接后形成 561 维向量。

例如：

- `tBodyAcc-mean()-X` 是一个窗口内身体加速度 X 轴的均值，只占 1 维；
- `tBodyAcc-std()-Z` 是同一窗口内身体加速度 Z 轴的标准差，也占 1 维；
- `fBodyGyro-bandsEnergy()-9,16.2` 是陀螺仪某一轴在指定 FFT bin 区间内的能量，同样占 1 维。

因此，模型接收到的每个样本不是原始的 `128 × 6` 矩阵，而是一个由该窗口计算出的 `561` 维人工特征向量。原始窗口仍保存在 `Inertial Signals/` 中，两种表示可以分别用于端到端时序模型和人工特征模型。

#### 561 维的总体组成

根据数据集自带的 `features.txt`，561 维可以精确拆分为：

| 索引范围 | 类型 | 维度 | 内容 |
| --- | --- | ---: | --- |
| 1～265 | 时域特征 | 265 | 对身体/重力加速度、jerk、角速度及其模长进行统计 |
| 266～554 | 频域特征 | 289 | 对部分身体运动信号做 FFT 后计算频谱统计量和频带能量 |
| 555～561 | 夹角特征 | 7 | 运动均值向量、重力向量及坐标轴之间的夹角 |
| **合计** |  | **561** | `265 + 289 + 7 = 561` |

#### 时域特征：265 维

五组三轴信号各产生 40 维，五组模长信号各产生 13 维：

| 索引范围 | 信号 | 维度 |
| --- | --- | ---: |
| 1～40 | `tBodyAcc-XYZ` | 40 |
| 41～80 | `tGravityAcc-XYZ` | 40 |
| 81～120 | `tBodyAccJerk-XYZ` | 40 |
| 121～160 | `tBodyGyro-XYZ` | 40 |
| 161～200 | `tBodyGyroJerk-XYZ` | 40 |
| 201～213 | `tBodyAccMag` | 13 |
| 214～226 | `tGravityAccMag` | 13 |
| 227～239 | `tBodyAccJerkMag` | 13 |
| 240～252 | `tBodyGyroMag` | 13 |
| 253～265 | `tBodyGyroJerkMag` | 13 |

其中每组三轴信号的 40 维来自：

```text
mean 3 + std 3 + mad 3 + max 3 + min 3
+ sma 1 + energy 3 + iqr 3 + entropy 3
+ arCoeff 12 + correlation 3
= 40
```

- 带数字 `3` 的统计量分别作用于 X、Y、Z 三个轴。
- `sma` 是三轴共同计算的 Signal Magnitude Area，因此只有 1 维。
- `arCoeff` 对三个轴分别计算四阶 Burg 自回归系数，因此是 `3 × 4 = 12` 维。
- `correlation` 是 XY、XZ、YZ 三组轴间相关系数，因此是 3 维。

每组模长信号已经把 XYZ 合成为单通道，因此其 13 维为：

```text
mean + std + mad + max + min + sma + energy
+ iqr + entropy + 4 个 arCoeff
= 13
```

所以时域部分共有：

```text
5 × 40 + 5 × 13 = 265 维
```

#### 频域特征：289 维

官方对部分身体加速度、jerk 和角速度信号执行 128 点 FFT。实值信号的频谱按 64 个频率 bin 组织，并从中计算频谱统计量，而不是把全部频谱值直接作为模型输入。

| 索引范围 | 信号 | 维度 |
| --- | --- | ---: |
| 266～344 | `fBodyAcc-XYZ` | 79 |
| 345～423 | `fBodyAccJerk-XYZ` | 79 |
| 424～502 | `fBodyGyro-XYZ` | 79 |
| 503～515 | `fBodyAccMag` | 13 |
| 516～528 | `fBodyBodyAccJerkMag` | 13 |
| 529～541 | `fBodyBodyGyroMag` | 13 |
| 542～554 | `fBodyBodyGyroJerkMag` | 13 |

`fBodyBody...` 是原始 `features.txt` 中保留的官方命名，并非本项目重复输入了两次 `Body`。

每组三轴频域信号的 79 维来自：

```text
mean 3 + std 3 + mad 3 + max 3 + min 3
+ sma 1 + energy 3 + iqr 3 + entropy 3
+ maxInds 3 + meanFreq 3 + skewness 3 + kurtosis 3
+ bandsEnergy 42
= 79
```

其中 `bandsEnergy` 对每个轴计算 14 个预定义 FFT bin 区间的能量，因此共有 `3 × 14 = 42` 维。每组单通道频域模长信号则计算 mean、std、mad、max、min、sma、energy、iqr、entropy、maxInds、meanFreq、skewness 和 kurtosis，共 13 维。

所以频域部分共有：

```text
3 × 79 + 4 × 13 = 289 维
```

#### 夹角特征：7 维

最后 7 维对应 `features.txt` 中的第 555～561 项：

1. 身体加速度均值向量与重力向量的夹角；
2. 身体加速度 jerk 均值向量与重力均值向量的夹角；
3. 身体角速度均值向量与重力均值向量的夹角；
4. 身体角速度 jerk 均值向量与重力均值向量的夹角；
5. X 轴与重力均值向量的夹角；
6. Y 轴与重力均值向量的夹角；
7. Z 轴与重力均值向量的夹角。

#### 特征统计量含义

官方使用的统计属性包括：

- 均值、标准差、中位数绝对偏差、最大值和最小值；
- Signal Magnitude Area、能量、四分位距和信号熵；
- 四阶 Burg 自回归系数、轴间相关系数；
- 最大频率分量索引、平均频率、偏度和峰度；
- FFT 频带能量以及向量夹角。

本项目不会直接将全部 561 维官方特征作为唯一输入，而是从窗口信号重新实现一组适合资源受限设备的精简时频域特征，并将官方特征结果作为参考基线。

### 关键文件

```text
data/raw/UCI HAR Dataset/
├─ activity_labels.txt       # 数字标签与活动名称
├─ features.txt              # 561 个官方特征名称
├─ features_info.txt         # 官方特征计算说明
├─ train/
│  ├─ X_train.txt            # 7,352 × 561 官方训练特征
│  ├─ y_train.txt            # 训练标签
│  ├─ subject_train.txt      # 每个窗口对应的受试者编号
│  └─ Inertial Signals/      # 9 组、每窗口 128 点的训练信号
└─ test/
   ├─ X_test.txt             # 2,947 × 561 官方测试特征
   ├─ y_test.txt             # 测试标签
   ├─ subject_test.txt       # 每个窗口对应的受试者编号
   └─ Inertial Signals/      # 9 组、每窗口 128 点的测试信号
```

### 使用限制与项目注意事项

- 数据来自腰部固定位置的手机，模型结果不能直接代表手腕、鞋部或自由放置设备上的表现。
- 六种活动均在受控实验环境中采集，真实开放环境可能存在更强的佩戴差异和类别过渡噪声。
- 相邻窗口有 50% 重叠，评估时必须保持受试者隔离，避免将高度相似的相邻窗口随机分配到训练集和测试集。
- 桌面端 TFLM 延迟只能作为 host benchmark，不能视为 MCU 实测延迟或功耗指标。

## 计划范围

- 使用 UCI HAR 六轴惯性数据完成六类人体活动识别。
- 实现 Python 与 C++ 时频域特征，并通过 golden vectors 验证一致性。
- 使用 PyTorch 训练轻量化 MLP，并生成全 INT8 TFLite 模型。
- 使用桌面端 TFLM `MicroInterpreter`、最小算子注册和静态 tensor arena 完成推理。
- 比较 FP32、INT8 与 TFLM 的精度、模型大小、内存需求和桌面端延迟。

## 边界说明

本项目当前不宣称已部署至真实 MCU，也不提供真实板端延迟、功耗或动态频率调节结果。桌面端测试结果将明确标注为 host simulation。

## 目录

```text
benchmark/          基准测试输出与说明
configs/            数据、训练、量化和推理配置
data/               原始与处理后数据（数据文件不纳入 Git）
docs/               设计、复现和面试说明
firmware/           面向 MCU 风格的 C++ 特征与 TFLM 推理代码
models/             模型文件（生成物不纳入 Git）
scripts/            下载、训练、转换和验证入口
tests/              单元测试与 golden vectors
training/           PyTorch 训练与模型转换代码
feature_ref/        Python 特征参考实现
artifacts/           图表、日志和演示产物（生成物不纳入 Git）
```
