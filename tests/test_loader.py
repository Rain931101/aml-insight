"""資料讀取的基本測試。需先執行 scripts/00_fetch_amlsim_samples.sh。"""

import pytest

from aml_insight.data.loader import DATASETS, LEAKAGE_COLUMNS, RAW_DIR, load_dataset

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / DATASETS[0] / "nodes.csv").exists(), reason="原始資料尚未下載"
)


@pytest.mark.parametrize("name", DATASETS)
def test_structure(name):
    ds = load_dataset(name)
    assert len(ds.nodes) == 20_000
    assert ds.nodes["nodeid"].is_unique
    ids = set(ds.nodes["nodeid"])
    assert ds.transactions["sourceNodeId"].isin(ids).all()
    assert ds.transactions["targetNodeId"].isin(ids).all()
    assert set(ds.nodes["isFraud"].unique()) <= {0, 1}


def test_fraudstep_is_one_way_leak():
    """非詐欺帳戶的 fraudStep 必為 -1 → fraudStep 不可作為特徵。"""
    ds = load_dataset("20K_fanin200cycle200")
    assert (ds.nodes.loc[ds.nodes.isFraud == 0, "fraudStep"] == -1).all()
    assert "fraudStep" in LEAKAGE_COLUMNS


def test_unknown_dataset():
    with pytest.raises(ValueError):
        load_dataset("not_a_dataset")
