"""轻量 MLP 的接口与资源规模测试。"""

import pytest
import torch

from feature_ref.feature_schema import FEATURE_DIM
from models.mlp import NUM_CLASSES, TinyMLP, count_trainable_parameters


def test_mlp_output_shape_and_parameter_count() -> None:
    model = TinyMLP()
    inputs = torch.zeros((4, FEATURE_DIM), dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (4, NUM_CLASSES)
    assert count_trainable_parameters(model) == 1_910


def test_mlp_accepts_single_feature_vector() -> None:
    model = TinyMLP()

    outputs = model(torch.zeros(FEATURE_DIM, dtype=torch.float32))

    assert outputs.shape == (NUM_CLASSES,)


def test_mlp_rejects_wrong_feature_dimension() -> None:
    model = TinyMLP()

    with pytest.raises(ValueError, match="expected final dimension 39"):
        model(torch.zeros((2, FEATURE_DIM - 1), dtype=torch.float32))


def test_mlp_rejects_schema_mismatch_at_construction() -> None:
    with pytest.raises(ValueError, match="input_dim must match feature schema"):
        TinyMLP(input_dim=FEATURE_DIM + 1)
