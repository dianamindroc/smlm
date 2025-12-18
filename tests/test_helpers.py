import numpy as np
import pandas as pd
import torch

from helpers import data as data_utils
from model_architectures.transforms import Padding, ToTensor


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("x,y,z\n")
        for x, y, z in rows:
            handle.write(f"{x},{y},{z}\n")


def test_assign_labels_creates_unique_integer_mapping(tmp_path):
    for name in ["cube", "pyramid", "tetra"]:
        (tmp_path / name).mkdir()

    labels = data_utils.assign_labels(str(tmp_path))

    assert set(labels.keys()) == {"cube", "pyramid", "tetra"}
    assert sorted(labels.values()) == list(range(len(labels)))


def test_get_highest_shape_counts_rows_without_header(tmp_path):
    for cls in ["cube", "pyramid"]:
        for sub in ["iso", "aniso"]:
            target_dir = tmp_path / cls / sub
            target_dir.mkdir(parents=True)

    _write_csv(tmp_path / "cube" / "iso" / "a.csv", [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    _write_csv(tmp_path / "pyramid" / "aniso" / "b.csv", [(0, 0, 0), (1, 1, 1)])

    highest = data_utils.get_highest_shape(str(tmp_path), classes=["cube", "pyramid"])

    assert highest == 3


def test_get_label_binary_mapping():
    assert data_utils.get_label("cube", ["cube", "pyramid"]) == 0
    assert data_utils.get_label("pyramid", ["cube", "pyramid"]) == 1


def test_add_anisotropy_scales_selected_axis():
    pc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stretched = data_utils.add_anisotropy(pc, anisotropy_factor=2.5, anisotropic_axis="z")

    np.testing.assert_allclose(stretched[:, :2], pc[:, :2])
    np.testing.assert_allclose(stretched[:, 2], pc[:, 2] * 2.5)


def test_get_bounding_box_size_handles_numpy_and_dataframe():
    arr = np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 5.0]])
    df = pd.DataFrame(arr, columns=["x", "y", "z"])

    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(data_utils.get_bounding_box_size(arr), expected)
    np.testing.assert_allclose(data_utils.get_bounding_box_size(df), expected)


def test_padding_pad_point_cloud_preserves_points_and_mask():
    np.random.seed(0)
    padder = Padding(highest_shape=5)
    pc = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    padded, mask = padder.pad_point_cloud(pc)

    assert padded.shape == (5, 3)
    np.testing.assert_allclose(padded[: pc.shape[0]], pc)
    assert mask.shape == (5,)
    np.testing.assert_allclose(mask[: pc.shape[0]], 1)
    np.testing.assert_allclose(mask[pc.shape[0] :], 0)


def test_padding_call_applies_to_all_fields():
    np.random.seed(1)
    padder = Padding(highest_shape=4)
    sample = {
        "pc": np.array([[0, 0, 0], [1, 1, 1]]),
        "partial_pc": np.array([[0, 0, 0]]),
        "pc_anisotropic": np.array([[1, 2, 3]]),
    }

    padded = padder(sample)

    assert padded["pc"].shape == (4, 3)
    assert padded["partial_pc"].shape == (4, 3)
    assert padded["pc_anisotropic"].shape == (4, 3)
    assert padded["pc_mask"].sum() == 2
    assert padded["partial_pc_mask"].sum() == 1
    assert padded["pc_anisotropic_mask"].sum() == 1


def test_to_tensor_converts_arrays_and_labels():
    sample = {
        "pc": np.array([[0.0, 0.0, 0.0]]),
        "partial_pc": np.array([[1.0, 1.0, 1.0]]),
        "pc_mask": np.array([1, 0]),
        "partial_pc_mask": np.array([1, 0]),
        "pc_anisotropic": np.array([[2.0, 2.0, 2.0]]),
        "label": 3,
    }

    converted = ToTensor()(sample)

    assert isinstance(converted["pc"], torch.Tensor)
    assert converted["pc"].dtype == torch.float32
    assert isinstance(converted["label"], torch.Tensor)
    torch.testing.assert_close(converted["label"], torch.tensor([3.0]))
