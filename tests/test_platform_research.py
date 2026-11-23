"""Research platform must replay sealed evidence without changing scientific meaning."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from research_data import (
    MODELS,
    precision_recall,
    predictions,
    read_json,
    run_path,
    verify_bundle,
)


def test_bundle_and_metric_replay():
    assert len(verify_bundle()["sources"]) == 10
    for setting in ["S1", "S2"]:
        for seed in range(5):
            pred = predictions(setting, seed)
            metrics = read_json(run_path(setting, seed) / "metrics.json")["models"]
            assert len(pred) == 4000
            assert pred.nodeid.is_unique
            for model in MODELS:
                assert average_precision_score(
                    pred.isFraud, pred[f"{model}_score"]
                ) == pytest.approx(metrics[model]["pr_auc"], abs=1e-7)


def test_pr_curve_groups_ties_and_matches_ap():
    y = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.8, 0.8, 0.4, 0.2, 0.2])
    recall, precision = precision_recall(y, scores)
    assert len(recall) == 4
    assert np.sum(np.diff(recall) * precision[1:]) == pytest.approx(
        average_precision_score(y, scores)
    )


@pytest.mark.parametrize(
    "page", ["研究總覽", "資料鑑識", "模型實驗", "帳戶與網絡", "解釋證據", "研究驗證", "方法與來源"]
)
def test_platform_pages(page):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app/Home.py", default_timeout=30)
    app.query_params["view"] = page
    app.run()
    assert not app.exception
    assert not app.error


def test_case_filter_and_dataset_scope():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app/Home.py", default_timeout=30)
    app.query_params["view"] = "帳戶與網絡"
    app.run()
    next(w for w in app.radio if w.label == "篩選案例").set_value("假陰性 FN").run()
    assert not app.exception
    selected_id = next(w for w in app.selectbox if w.label == "搜尋或選取帳戶 ID").value
    pred = predictions("S2", 0).set_index("nodeid").loc[selected_id]
    assert pred.isFraud == 1 and pred.graphsage_pred == 0
    next(w for w in app.selectbox if w.label == "資料集").set_value("20K_cycle200").run()
    assert not app.exception
    assert any("尚無已封存" in title.value for title in app.title)


def test_navigation_persists_after_rerun():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app/Home.py", default_timeout=30
    ).run()
    next(w for w in app.radio if w.label == "工作區").set_value("模型實驗").run()
    app.run()
    assert not app.exception
    assert app.session_state["workspace"] == "模型實驗"
    assert any("最高分" in title.value for title in app.title)


def test_follow_up_research_views():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/Home.py', default_timeout=30)
    app.query_params['view'] = '研究驗證'
    app.run()
    assert not app.exception and not app.error
    assert app.metric[0].value == '85'
    for section in ['外部資料', '解釋可靠性', '完整報告']:
        next(w for w in app.radio if w.label == '研究問題').set_value(section).run()
        assert not app.exception and not app.error
