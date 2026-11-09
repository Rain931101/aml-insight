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
