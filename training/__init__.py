"""UCI HAR 数据加载、预处理、训练与评估工具。"""

from .data import (
    ACTIVITY_NAMES,
    SIGNAL_CHANNELS,
    UCIHARDataset,
    create_data_loaders,
    iter_signal_dicts,
    load_uci_har_split,
    split_dataset_by_subject,
)

__all__ = [
    "ACTIVITY_NAMES",
    "SIGNAL_CHANNELS",
    "UCIHARDataset",
    "create_data_loaders",
    "iter_signal_dicts",
    "load_uci_har_split",
    "split_dataset_by_subject",
]
