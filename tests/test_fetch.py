"""成員 C：雲端資料下載（不連網，只測「已有資料就不下載」）。"""

from aml_insight.data import fetch


def test_ensure_samples_skips_existing(tmp_path, monkeypatch):
    for name in fetch.DATASETS:
        (tmp_path / name).mkdir()
        (tmp_path / name / "nodes.csv").write_text("nodeid\n")

    def no_network(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", no_network)
    assert fetch.ensure_samples(tmp_path) == []
    assert (tmp_path / "AMLSIM_COMMIT.txt").read_text().strip() == fetch.AMLSIM_COMMIT
