# Models

模型代码和本地生成物目录：

- `mlp.py`：基于 39 维精简 IMU 特征的 PyTorch 轻量 MLP；
- `checkpoints/`：训练检查点；
- `exported/`：转换后的 Keras、FP32 TFLite 和全 INT8 TFLite 模型。

`mlp.py` 提交到 Git；检查点和导出文件由 `.gitignore` 排除。用于桌面 TFLM 验证的 INT8
模型以 C++ 字节数组形式保存在 `firmware/model_data.cc`。
