import importlib
import sys
from dataclasses import dataclass
from typing import Any, Dict

import pytest
import torch


# Helpers ---------------------------------------------------------------------
def _fake_args(argv):
    """Replace sys.argv for CLI tests."""
    sys.argv = argv


@dataclass
class RecordedCall:
    kwargs: Dict[str, Any]

    def __init__(self):
        self.kwargs = {}

    def record(self, **kwargs):
        self.kwargs.update(kwargs)


# run_training ----------------------------------------------------------------
def test_run_training_cli_invokes_train(monkeypatch, tmp_path):
    from scripts import run_training

    called = RecordedCall()

    def fake_train(config, exp_name=None, fixed_alpha=None):
        called.record(config=config, exp_name=exp_name, fixed_alpha=fixed_alpha)

    monkeypatch.setattr(run_training, "train", fake_train)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dummy: true", encoding="utf-8")

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    _fake_args(["run_training.py", "--config", str(cfg), "--exp_name", "unit", "--fixed_alpha", "0.5"])

    run_training.main()

    assert called.kwargs["config"] == str(cfg)
    assert called.kwargs["exp_name"] == "unit"
    assert called.kwargs["fixed_alpha"] == 0.5


# simulate_data ---------------------------------------------------------------
def test_simulate_data_cli(monkeypatch):
    from scripts import simulate_data
    from simulation import simulate_dna_origami

    init = RecordedCall()
    generated = RecordedCall()

    def fake_init(self, struct_type, number_dna_origami_samples, stats=None, apply_rotation=True):
        init.record(
            struct_type=struct_type,
            number=number_dna_origami_samples,
            stats=stats,
            apply_rotation=apply_rotation,
        )

    def fake_generate(self):
        generated.record(called=True)

    monkeypatch.setattr(simulate_dna_origami.SMLMDnaOrigami, "__init__", fake_init)
    monkeypatch.setattr(simulate_dna_origami.SMLMDnaOrigami, "generate_all_dna_origami_smlm_samples", fake_generate)
    _fake_args(["simulate_data.py", "-s", "tetrahedron", "-n", "1"])

    simulate_data.main()

    assert init.kwargs == {
        "struct_type": "tetrahedron",
        "number": 1,
        "stats": None,
        "apply_rotation": False,
    }
    assert generated.kwargs.get("called") is True


# test_pocafoldas -------------------------------------------------------------
def test_test_pocafoldas_imports():
    mod = importlib.import_module("scripts.test_pocafoldas")
    mod = importlib.reload(mod)
    assert hasattr(mod, "load_model")


def test_test_single_category_smoke(monkeypatch, tmp_path):
    from scripts import test_pocafoldas

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.data = [
                {
                    "partial_pc": torch.zeros(3, 3),
                    "pc": torch.ones(3, 3),
                    "filename": "a.ply",
                    "label": torch.tensor([0]),
                    "label_name": "cls",
                    "corner_label": torch.tensor([0]),
                    "corner_label_name": "corner",
                },
                {
                    "partial_pc": torch.zeros(3, 3),
                    "pc": torch.ones(3, 3),
                    "filename": "b.ply",
                    "label": torch.tensor([1]),
                    "label_name": "cls",
                    "corner_label": torch.tensor([0]),
                    "corner_label_name": "corner",
                },
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    class FakeModel:
        def __call__(self, _x):
            output = torch.zeros((1, 3, 3))
            feature = torch.zeros((1, 4))
            logits = torch.zeros((1, 2))
            attn = torch.zeros((1, 1))
            return None, output, feature, logits, attn

    monkeypatch.setattr(test_pocafoldas, "PairedAnisoIsoDataset", FakeDataset)
    monkeypatch.setattr(test_pocafoldas, "get_highest_shape", lambda *_, **__: 3)
    monkeypatch.setattr(test_pocafoldas, "export_ply", lambda *_, **__: None)
    monkeypatch.setattr(test_pocafoldas, "l1_cd", lambda *_: torch.tensor(0.1))
    monkeypatch.setattr(test_pocafoldas, "l2_cd", lambda *_: torch.tensor(0.2))

    test_config = {
        "root_folder": "demo",
        "suffix": ".csv",
        "classes_to_use": ["cls"],
        "remove_part_prob": 0.0,
        "remove_corners": False,
        "remove_outliers": False,
        "num_corners_remove": [0],
    }

    results = test_pocafoldas.test_single_category(
        FakeModel(),
        device=torch.device("cpu"),
        test_config=test_config,
        log_dir=str(tmp_path),
        save=True,
    )

    assert results["avg_l1_cd"] == pytest.approx(0.1)
    assert results["avg_l2_cd"] == pytest.approx(0.2)
    assert results["predicted_classes"].shape[0] == len(FakeDataset())
