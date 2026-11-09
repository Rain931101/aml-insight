import pytest
import yaml


def test_suite_generates_s1_s2_seeds_and_ablations(tmp_path):
    from aml_insight.evaluation.suite import generate_suite

    cfg = {
        "features": "features.csv",
        "run_id": "base",
        "experiment": "E1",
        "seed": 0,
        "setting": "S1",
        "output_root": "runs",
        "feature_spec": {
            "groups": {"amount": ["amount"], "behavior": ["count"], "structure": ["degree"]}
        },
        "rules": {
            "S1": [{"feature": "count", "direction": "high"}],
            "S2": [{"feature": "count", "direction": "high"}],
        },
    }
    p = tmp_path / "base.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out = generate_suite(p, tmp_path / "suite")
    files = sorted(out.glob("*.yaml"))
    assert (
        len(files) == 30
    )  # E1 10 + E2 behavior/structure 20; E5 reuses E1; amount removal equals S2.
    assert all(
        yaml.safe_load(f.read_text())["features"] == str(tmp_path / "features.csv") for f in files
    )
    assert (out / "pending.json").exists()
    with pytest.raises(FileExistsError):
        generate_suite(p, out)


def test_temporal_suite_uses_distinct_full_and_early_features(tmp_path):
    from aml_insight.evaluation.suite import generate_suite

    cfg = {
        "features": "full.parquet",
        "time_train_features": "early.parquet",
        "time_test_features": "late.parquet",
        "run_id": "temporal",
        "experiment": "E1",
        "seed": 0,
        "setting": "S1",
        "feature_spec": {"groups": {"amount": ["amount"], "behavior": ["count"]}},
        "time": {
            "train_start": 1,
            "train_end": 89,
            "test_start": 90,
            "test_end": 149,
            "feature_observation_attestation": "test fixture only",
        },
    }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = generate_suite(path, tmp_path / "suite")
    generated = [yaml.safe_load(p.read_text()) for p in out.glob("*.yaml")]
    baseline = [c for c in generated if c["experiment"] == "E1"]
    temporal = [c for c in generated if c["experiment"] == "E3"]
    assert len(baseline) == len(temporal) == 10
    assert all(c["features"] == str(tmp_path / "full.parquet") for c in baseline)
    assert all(c["features"] == str(tmp_path / "early.parquet") for c in temporal)
    assert all(c["time_test_features"] == str(tmp_path / "late.parquet") for c in temporal)


@pytest.mark.parametrize("early", [None, "full.parquet"])
def test_temporal_suite_rejects_missing_or_reused_early_path(tmp_path, early):
    from aml_insight.evaluation.suite import generate_suite

    cfg = {
        "features": "full.parquet",
        "time_test_features": "late.parquet",
        "run_id": "temporal",
        "experiment": "E1",
        "seed": 0,
        "setting": "S1",
        "feature_spec": {"groups": {"amount": [], "behavior": ["count"]}},
    }
    if early:
        cfg["time_train_features"] = early
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="time_train_features"):
        generate_suite(path, tmp_path / "suite")
    assert not (tmp_path / "suite").exists()
