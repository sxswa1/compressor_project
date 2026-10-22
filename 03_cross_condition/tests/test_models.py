"""模型层冒烟测试：验证模型工厂与域自适应模型可实例化并前向。"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT_03 = HERE.parent
PROJ = ROOT_03.parent
sys.path.insert(0, str(ROOT_03))
sys.path.insert(0, str(PROJ))

from models.domain_adaptation import create_model, list_models  # noqa: E402
from models.feature_alignment import physics_features  # noqa: E402


def test_list_models():
    names = list_models()
    assert "src_wdcnn" in names
    assert "dann_wdcnn" in names
    assert "coral_wdcnn" in names
    assert "dan_wdcnn" in names


def test_forward():
    x = torch.randn(4, 1, 1024)
    for name in ["src_wdcnn", "dann_wdcnn", "coral_wdcnn", "dan_wdcnn", "src_lstm"]:
        model = create_model(name, num_classes=10)
        out = model(x) if not name.startswith("dann") else model(x, alpha=0.5)
        assert out["logits"].shape == (4, 10)
        assert out["feats"].shape[1] == 256


def test_physics_features_shape():
    import numpy as np
    x = np.random.default_rng(0).standard_normal((4, 1024)).astype(np.float32)
    f = physics_features(x, 12000, 1797, n_harmonics=3)
    assert f.shape == (4, 4 * 3)  # 4 阶次 × 3 谐波


if __name__ == "__main__":
    for fn in [test_list_models, test_forward, test_physics_features_shape]:
        fn()
        print(f"{fn.__name__} OK")
