# Data

数据集和处理结果保存在本地，不提交到 Git；仓库只保留目录说明。

- `raw/`：原始 UCI HAR 数据。
- `processed/`：由 `python -m training.preprocess` 生成的特征、标签、标准化参数和划分元数据。
