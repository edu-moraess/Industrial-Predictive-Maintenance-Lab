"""Dataset factory smoke tests."""

from pathlib import Path

from dataset.generator import generate_dataset, render_machine
from dataset.validator import validate


def test_render_machine_normal():
    img, anns, meta = render_machine("normal", 0.0, seed=1)
    assert img.size[0] > 0
    assert meta["synthetic"] is True
    assert any(a["class"] == "motor" for a in anns)


def test_generate_tiny_dataset(tmp_path: Path):
    out = generate_dataset(n_per_condition=2, seed=0, out_root=tmp_path / "ds")
    assert (out / "dataset_manifest.csv").exists()
    rep = validate(out)
    assert rep["ok"], rep["errors"]
    assert rep["stats"]["n_images"] == 2 * 7  # 7 conditions
