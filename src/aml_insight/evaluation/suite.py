"""Generate reviewable experiment configs; tuning and test execution stay separate."""

from copy import deepcopy
from pathlib import Path

import yaml

from aml_insight.data.split import SEEDS
from aml_insight.evaluation.artifacts import write_json
from aml_insight.evaluation.runner import load_config, resolve


def generate_suite(config_path, output):
    base_cfg, base = load_config(config_path)
    if base_cfg.get("time_test_features"):
        early = base_cfg.get("time_train_features")
        if not early or resolve(base, early) == resolve(base, base_cfg["features"]):
            raise ValueError(
                "Temporal suite requires separate time_train_features for early-window training; "
                "features remains the full-window E1 baseline"
            )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for key in (
        "features",
        "time_train_features",
        "time_test_features",
        "analysis_metadata",
        "output_root",
    ):
        if base_cfg.get(key):
            base_cfg[key] = str(resolve(base, base_cfg[key]))
    base_cfg.setdefault("output_root", str(resolve(base, "../experiments")))
    # C must train on each exact split/variant. Never reuse one prediction file across seeds.
    for key in ("graphsage", "split_manifest"):
        base_cfg.pop(key, None)
    groups = base_cfg["feature_spec"]["groups"]
    variants = [("E1", [], True)]
    # Amount removal is already S2; avoid duplicate final test evaluation.
    variants += [
        ("E2", [group], False) for group, cols in groups.items() if group != "amount" and cols
    ]
    if base_cfg.get("time_test_features"):
        variants += [("E3", [], True)]
    count = 0
    for experiment, dropped, rules in variants:
        for setting in ("S1", "S2"):
            selected = [
                c
                for g, cols in groups.items()
                if g not in dropped and not (setting == "S2" and g == "amount")
                for c in cols
            ]
            if not selected:
                continue
            for seed in SEEDS:
                cfg = deepcopy(base_cfg)
                if experiment == "E3":
                    cfg["features"] = base_cfg["time_train_features"]
                # These are suite inputs only; each emitted run has one explicit training table.
                cfg.pop("time_train_features", None)
                suffix = "-without-" + "-".join(dropped) if dropped else ""
                cfg.update(
                    experiment=experiment,
                    setting=setting,
                    seed=seed,
                    drop_groups=dropped,
                    include_rules=rules,
                    run_id=f"{base_cfg['run_id']}-{experiment}-{setting}-seed{seed}{suffix}",
                )
                (output / f"{cfg['run_id']}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                count += 1
    write_json(
        output / "pending.json",
        {
            "configs_generated": count,
            "E4": "Explainable AI case analysis: XGBoost + SHAP available; pending C GraphSAGE + GNNExplainer and representative matched cases",
            "E5": "Aggregate E1 seed 0..4 artifacts; do not rerun the same test split",
            "E2_graph": "Pending A controlled graph variants and C GraphSAGE results; generated configs cover feature ablation only",
            "E2_amount": "S1 without amount equals S2; compare E1 S1/S2 instead of rerunning",
            "E2_models": "XGBoost only; ablated rule features may no longer exist",
            "E3": "Separate full/early/late feature paths configured"
            if base_cfg.get("time_test_features")
            else "Pending A time features",
            "GraphSAGE": "Attach C metadata and validation/test files for each exact variant",
        },
    )
    return output
