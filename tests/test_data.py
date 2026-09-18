import csv

import numpy as np

from u3dgpr.data import GPRVolumeDataset, normalize_gpr


def test_fixed_minmax_matches_official_3dinvnet_loader():
    source = np.asarray([-9.0, 0.0, 9.0], dtype=np.float32)
    normalized = normalize_gpr(source, "fixed_minmax", input_min=-9.0, input_max=9.0)
    np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0])


def test_axis_order_is_applied_before_cubic_shape_short_circuit(tmp_path):
    source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    input_path = tmp_path / "input.npy"
    target_path = tmp_path / "target.npy"
    np.save(input_path, source)
    np.save(target_path, source + 100)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "input", "target"))
        writer.writeheader()
        writer.writerow({"id": "one", "input": input_path, "target": target_path})

    dataset = GPRVolumeDataset(
        manifest,
        shape=(3, 4, 2),
        input_normalization="none",
        axis_order=(1, 2, 0),
    )
    sample = dataset[0]
    np.testing.assert_array_equal(sample["gpr"][0].numpy(), source.transpose(1, 2, 0))
    np.testing.assert_array_equal(sample["target"][0].numpy(), (source + 100).transpose(1, 2, 0))
