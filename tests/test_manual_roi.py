from pathlib import Path

from vision.manual_roi import load_polygon, save_polygon


def test_save_load_roundtrip(tmp_path: Path):
    poly = [(10, 20), (300, 22), (310, 250), (8, 240)]
    path = tmp_path / "roi.json"
    save_polygon(path, poly)
    assert load_polygon(path) == poly


def test_load_missing_returns_none(tmp_path: Path):
    assert load_polygon(tmp_path / "nope.json") is None


def test_load_invalid_json_returns_none(tmp_path: Path):
    path = tmp_path / "roi.json"
    path.write_text("not json{")
    assert load_polygon(path) is None


def test_load_wrong_length_returns_none(tmp_path: Path):
    path = tmp_path / "roi.json"
    path.write_text("[[1, 2], [3, 4], [5, 6]]")  # 3점뿐
    assert load_polygon(path) is None
