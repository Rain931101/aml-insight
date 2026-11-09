"""Artifact-backed plots, error analysis and seed summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from aml_insight.evaluation.artifacts import read_json, verify, write_json


def plot_pr(y, scores, output):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.step(r, p, where="post", label=f"{name} (AP={average_precision_score(y, s):.4f})")
    ax.axhline(np.mean(y), color="gray", linestyle="--", label="prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_error_analysis(run, predictions):
    names = [c.removesuffix("_pred") for c in predictions if c.endswith("_pred")]
    missed = predictions.isFraud.eq(1)
    for name in names:
        missed &= predictions[f"{name}_pred"].eq(0)
    predictions.loc[missed].to_csv(run / "common_false_negatives.csv", index=False)
    results = {
        "included_models": names,
        "common_false_negatives": int(missed.sum()),
        "comparison_complete": "graphsage" in names,
        "limitation": "Common misses include only supplied models; absence of GraphSAGE "
        "does not support a three-model conclusion.",
    }
    meta = run / "analysis_metadata.csv"
    if meta.exists():
        md = pd.read_csv(meta).set_index("nodeid").loc[predictions.nodeid]
        subset = predictions.isFraud.eq(1).to_numpy() & md.has_small_amount.eq(0).to_numpy()
        results["no_small_amount_positive_count"] = int(subset.sum())
        results["no_small_amount_recall"] = {
            name: float(predictions.loc[subset, f"{name}_pred"].mean()) if subset.any() else None
            for name in names
        }
    else:
        results["no_small_amount_recall"] = None
        results["pending"] = "Member A analysis_metadata: nodeid,has_small_amount required"
    write_json(run / "error_analysis.json", results)
    lines = [
        "# 錯誤案例分析",
        "",
        f"納入模型：{', '.join(names)}。",
        f"共同漏判正樣本數：{int(missed.sum())}。",
        "",
        "GraphSAGE 尚未提供，三模型共同漏判分析 pending。"
        if "graphsage" not in names
        else "已納入三個模型。",
        "",
        "逐帳戶結果：common_false_negatives.csv；子群分析：error_analysis.json。",
    ]
    (run / "error_analysis.md").write_text("\n".join(lines) + "\n")


def _group_key(cfg, run, model=None):
    # Seeds may change, but experiment meaning, data bytes and selection strategy may not.
    keys = [
        "experiment",
        "setting",
        "feature_spec",
        "drop_groups",
        "time",
        "xgb_grid",
        "rules",
        "rule_quantile",
        "include_rules",
        "purpose",
    ]
    identity = {k: cfg.get(k) for k in keys}
    if (run / "provenance.json").exists():
        prov = read_json(run / "provenance.json")
        identity["data_sha256"] = prov.get("feature_file", {}).get("sha256")
        identity["time_sha256"] = prov.get("time_feature_file", {}).get("sha256")
        identity["source_sha256"] = prov.get("source_sha256")
        identity["versions"] = prov.get("versions")
    if model == "graphsage":
        c_meta = read_json(run / "graphsage_metadata.json")
        training = dict(c_meta.get("training_config", {}))
        for key in ("seed", "random_state"):
            training.pop(key, None)
        identity["graphsage"] = {
            "training_config": training,
            "feature_version": c_meta.get("feature_version"),
            "selected_columns": c_meta.get("selected_columns"),
        }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def _flatten(metrics):
    result = {k: metrics.get(k) for k in ("pr_auc", "roc_auc", "f1")}
    for k, values in metrics.get("top_k", {}).items():
        if k not in ("100", "500"):
            continue
        for stat in ("precision", "recall"):
            result[f"{stat}_at_{k}"] = values[stat]
    positive_k = metrics.get("top_k_positive_count", {})
    for stat in ("precision", "recall"):
        result[f"{stat}_at_positive_count"] = positive_k.get(stat)
    return result


def aggregate(root, output, include_test=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    records, sources = [], []
    for path in sorted(root.rglob("metrics.json")):
        run = path.parent
        if not (run / "COMPLETE.json").exists() or not (run / "config.json").exists():
            continue
        cfg, metrics = read_json(run / "config.json"), read_json(path)
        if not include_test and cfg.get("purpose", "research") != "research":
            continue
        verify(run, check_source=False)
        verify(run, check_source=False, manifest="result_integrity.json")
        if any(metrics.get(k) != cfg.get(k) for k in ("seed", "setting", "experiment")):
            raise ValueError("Metric/config identity integrity mismatch")
        for model, m in metrics["models"].items():
            identity = _group_key(cfg, run, model)
            records.append(
                {
                    "group": identity,
                    "experiment": cfg["experiment"],
                    "setting": cfg["setting"],
                    "model": model,
                    "seed": cfg["seed"],
                    "run": str(path),
                    **_flatten(m),
                }
            )
        sources.append(
            {
                "path": str(path),
                "experiment": cfg["experiment"],
                "setting": cfg["setting"],
                "seed": cfg["seed"],
            }
        )
    rows = []
    frame = pd.DataFrame(records)
    if len(frame):
        for key, group in frame.groupby(["group", "experiment", "setting", "model"]):
            if group.seed.duplicated().any():
                raise ValueError("Duplicate seed in comparable group; select one run per seed")
            row = dict(zip(["group", "experiment", "setting", "model"], key))
            row["n_runs"] = len(group)
            row["seeds"] = ",".join(map(str, sorted(group.seed.tolist())))
            row["five_seeds_complete"] = set(group.seed) == {0, 1, 2, 3, 4}
            metric_cols = [c for c in group if c in ("pr_auc", "roc_auc", "f1") or "_at_" in c]
            for col in metric_cols:
                v = pd.to_numeric(group[col], errors="coerce").dropna()
                row[f"{col}_mean"] = float(v.mean()) if len(v) else None
                row[f"{col}_std"] = float(v.std(ddof=1)) if len(v) > 1 else None
                row[f"{col}_best"] = float(v.max()) if len(v) else None
                row[f"{col}_n_defined"] = len(v)
            rows.append(row)
    summary = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    write_json(output / "sources.json", sources)
    lines = [
        "# 實驗與結果",
        "",
        "僅收錄完成的 run；PR-AUC 指 Average Precision，標準差為樣本標準差（ddof=1）。",
        "結論僅適用於目前 AMLSim 合成資料；模型分數不代表犯罪證據。",
        "",
        "同 group 才可合併 seeds；特徵、資料、程式及選參策略改變即分組。",
        "",
        "| 實驗 | 設定 | 模型 | seeds | AP 平均 ± SD | 最佳 AP | 五 seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    if include_test:
        lines.insert(2, "**驗證模式：可能含 purpose=test 合成測試；不得作為正式研究數據。**")
    for row in rows:
        mean, std = row.get("pr_auc_mean"), row.get("pr_auc_std")
        ap = f"{mean:.4f}" if mean is not None else "undefined"
        ap += f" ± {std:.4f}" if std is not None else " ± undefined"
        best = row.get("pr_auc_best")
        best_text = f"{best:.4f}" if best is not None else "undefined"
        lines.append(
            f"|{row['experiment']}|{row['setting']}|{row['model']}|{row['seeds']}|"
            f"{ap}|{best_text}|{'完整' if row['five_seeds_complete'] else 'pending'}|"
        )
    if not rows:
        lines += ["", "尚無正式研究結果，pending A 的特徵及正式執行。"]
    lines += ["", "## 研究問題與交付狀態", ""]
    for exp, description in [
        ("E1", "S1/S2 三模型比較"),
        ("E2", "特徵消融"),
        ("E3", "時間視窗轉移與隨機切分比較"),
        ("E5", "五 seeds 穩定性"),
    ]:
        relevant = [
            r for r in rows if r["experiment"] == exp or (exp == "E5" and r["experiment"] == "E1")
        ]
        lines.append(
            f"- {exp} {description}："
            + ("有部分 run，需依下列來源檢視設定完整性。" if relevant else "pending。")
        )
    lines += [
        "- E4：原始要求未定義，pending 規格。",
        "- GraphSAGE、時間特徵、特徵消融及無小額交易子群：依 A／C 輸入到齊程度判讀。",
        "- SHAP 與忠實度檔位於各 run/shap/；未產生時不可宣稱完成解釋。",
        "",
        "E5 直接引用 E1 的五 seeds，避免重複測試；只有完整五 seeds 才標為完整。",
        "",
        "## 數字來源",
        "",
    ]
    for item in sources:
        path = Path(item["path"])
        lines.append(
            f"- {item['experiment']} / {item['setting']} / seed {item['seed']}: "
            f"[{path.parent.name}/metrics.json]({path.as_uri()})"
        )
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return summary
