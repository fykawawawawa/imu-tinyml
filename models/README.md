# Models

模型定义和生成物目录：

- `mlp.py`：基于 39 维精简 IMU 特征的 PyTorch 轻量 MLP；
- `checkpoints/`：训练检查点；
- `exported/`：转换后的 Keras、TFLite 等模型。

模型源码提交到 Git，生成的权重和导出文件由 `.gitignore` 排除。
