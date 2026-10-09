# 桌面 TFLM 推理延迟

测量日期：2026-10-02

## 环境与口径

- CPU：AMD Ryzen 7 7735H（8 核 16 线程）
- 运行环境：Docker Desktop Linux 容器，固定到容器 CPU 0
- 编译器：GCC 12.2.0，`-O2 -std=c++17`
- TFLM 提交：`cd32a519356a05615e58001637ee25f73b009c86`
- INT8 模型 SHA256：`e48a80d16c67c13c9f3c6d15f096ab405c87b5a9f849bb573e788d86b7074d75`
- 每轮预热 1,000 次、测量 50,000 次，共独立运行 5 轮
- 六类 Golden Vector 循环作为输入；计时循环内不打印日志
- `Invoke-only` 仅包含 `MicroInterpreter::Invoke()`
- `End-to-end` 包含 39 维已标准化浮点特征检查、INT8 量化、`Invoke()`、输出反量化和
  `argmax`，不包含原始 IMU 窗口的特征提取及标准化

`steady_clock` 连续两次取时的 P50 开销为 0.020 us，相对推理延迟约为 1.5%。以下数据
未扣除计时器开销。

## 五轮结果

单位均为微秒。

| 轮次 | Invoke mean | Invoke P50 | Invoke P95 | Invoke P99 | 端到端 mean | 端到端 P50 | 端到端 P95 | 端到端 P99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.247 | 1.183 | 1.202 | 2.375 | 1.327 | 1.302 | 1.313 | 1.643 |
| 2 | 1.224 | 1.182 | 1.193 | 2.364 | 1.384 | 1.302 | 1.964 | 2.615 |
| 3 | 1.253 | 1.182 | 1.203 | 2.365 | 1.345 | 1.302 | 1.313 | 2.594 |
| 4 | 1.239 | 1.182 | 1.193 | 2.365 | 1.377 | 1.302 | 1.323 | 2.615 |
| 5 | 1.210 | 1.182 | 1.193 | 1.423 | 1.359 | 1.302 | 1.323 | 2.595 |
| **跨轮中位数** | **1.239** | **1.182** | **1.193** | **2.365** | **1.359** | **1.302** | **1.323** | **2.595** |

换算为毫秒后，纯推理 P50 为 **0.001182 ms**，当前特征级端到端 P50 为
**0.001302 ms**。极端最大值会受到桌面操作系统和容器调度影响，因此项目比较以 P50、
P95 和 P99 为主，不把单次最大值视为模型确定性执行时间。

## 复现

构建并运行一次：

```powershell
docker compose exec tflm bash -lc "cd /workspace && make -C firmware benchmark-latency"
```

固定到容器 CPU 0 重复五轮：

```powershell
docker compose exec tflm bash -lc `
  'for run in 1 2 3 4 5; do echo RUN=$run; taskset -c 0 /workspace/build/tflm-host/imu_tflm_latency_benchmark; done'
```

以上仅为 host simulation 结果，不能替代目标 MCU 上通过 cycle counter、硬件定时器或
GPIO 逻辑分析仪获得的板端延迟。
