"""精简时域特征参考实现的单元测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from feature_ref import FEATURE_DIM, FEATURE_INDEX, FEATURE_NAMES, extract_features
from feature_ref.time_features import safe_correlation


class TimeFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        # 构造包含线性相关、负相关和恒定通道的确定性测试窗口。
        samples = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
        self.signals = {
            "body_acc_x": samples,
            "body_acc_y": 2.0 * samples,
            "body_acc_z": -samples,
            "body_gyro_x": np.ones(128),
            "body_gyro_y": np.zeros(128),
            "body_gyro_z": np.full(128, -2.0),
            "total_acc_x": samples + 0.1,
            "total_acc_y": 2.0 * samples - 0.2,
            "total_acc_z": -samples + 0.9,
        }

    def test_schema_is_fixed_at_39_features(self) -> None:
        # schema 的维度、名称唯一性以及首尾索引都必须固定。
        self.assertEqual(FEATURE_DIM, 39)
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertEqual(FEATURE_INDEX["body_acc_x_mean"], 0)
        self.assertEqual(FEATURE_INDEX["body_gyro_corr_yz"], 38)

    def test_extracts_expected_shape_dtype_and_values(self) -> None:
        # 同时检查输出接口、重力恢复、相关系数及数值有限性。
        features = extract_features(self.signals)

        self.assertEqual(features.shape, (39,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertAlmostEqual(features[FEATURE_INDEX["gravity_mean_x"]], 0.1, places=6)
        self.assertAlmostEqual(features[FEATURE_INDEX["gravity_mean_y"]], -0.2, places=6)
        self.assertAlmostEqual(features[FEATURE_INDEX["gravity_mean_z"]], 0.9, places=6)
        self.assertAlmostEqual(features[FEATURE_INDEX["body_acc_corr_xy"]], 1.0)
        self.assertAlmostEqual(features[FEATURE_INDEX["body_acc_corr_xz"]], -1.0)
        self.assertAlmostEqual(features[FEATURE_INDEX["body_gyro_corr_xy"]], 0.0)

    def test_constant_signal_correlation_is_zero(self) -> None:
        # 恒定信号的相关系数按项目约定返回 0，而不是 NaN。
        self.assertEqual(safe_correlation(np.ones(128), np.arange(128)), 0.0)

    def test_rejects_missing_or_wrong_length_channels(self) -> None:
        # 缺少通道或不是 128 点的窗口都应尽早报错。
        missing = dict(self.signals)
        del missing["total_acc_z"]
        with self.assertRaises(KeyError):
            extract_features(missing)

        wrong_length = dict(self.signals)
        wrong_length["body_acc_x"] = np.zeros(127)
        with self.assertRaises(ValueError):
            extract_features(wrong_length)

    def test_rejects_non_finite_input(self) -> None:
        # NaN/Inf 不允许进入模型特征。
        invalid = dict(self.signals)
        invalid["body_acc_x"] = np.full(128, np.nan)
        with self.assertRaises(ValueError):
            extract_features(invalid)

    def test_first_real_uci_har_window_when_dataset_is_available(self) -> None:
        """读取 9 个训练文件的第一行，执行一次真实窗口端到端验证。"""
        signal_dir = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "raw"
            / "UCI HAR Dataset"
            / "train"
            / "Inertial Signals"
        )
        channel_names = (
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
        )
        if not signal_dir.is_dir():
            self.skipTest("本地未安装 UCI HAR 原始数据集")

        # 每个文件的第 i 行属于同一个窗口；这里统一读取第 1 行。
        signals = {}
        for name in channel_names:
            file_path = signal_dir / f"{name}_train.txt"
            with file_path.open("r", encoding="utf-8") as stream:
                signals[name] = np.fromstring(stream.readline(), sep=" ")

        features = extract_features(signals)

        self.assertEqual(features.shape, (FEATURE_DIM,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))


if __name__ == "__main__":
    unittest.main()
