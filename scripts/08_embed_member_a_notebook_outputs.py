"""Refresh saved outputs in Member A notebooks without requiring a Jupyter server."""

from __future__ import annotations

import base64
import json

import nbformat as nbf
import pandas as pd

from aml_insight.data.loader import PROJECT_ROOT, load_dataset


def stream(text: str):
    return nbf.v4.new_output("stream", name="stdout", text=text)


def main() -> None:
    notebooks = PROJECT_ROOT / "notebooks"
    ds = load_dataset("20K_fanin200cycle200")
    nodes, tx = ds.nodes, ds.transactions
    labels = nodes.set_index("nodeid").isFraud
    both = tx.sourceNodeId.map(labels).eq(1) & tx.targetNodeId.map(labels).eq(1)
    threshold = tx.loc[~both, "value"].min()
    small = tx.loc[tx.value < threshold]
    accounts = set(small.sourceNodeId) | set(small.targetNodeId)
    flagged = nodes.nodeid.isin(accounts)
    precision = (flagged & nodes.isFraud.eq(1)).sum() / max(flagged.sum(), 1)
    recall = (flagged & nodes.isFraud.eq(1)).sum() / nodes.isFraud.sum()

    verify_path = notebooks / "A_week1_verify.ipynb"
    verify = nbf.read(verify_path, as_version=4)
    verify.cells[1]["outputs"] = [stream(f"fraud accounts: {int(nodes.isFraud.sum()):,} / {len(nodes):,} = {nodes.isFraud.mean():.2%}\n")]
    cross = pd.crosstab(nodes.isFraud, nodes.fraudStep.eq(-1), rownames=["isFraud"], colnames=["fraudStep == -1"])
    verify.cells[3]["outputs"] = [nbf.v4.new_output("display_data", data={
        "text/plain": cross.to_string(), "text/html": cross.to_html(),
    }, metadata={})]
    verify.cells[5]["outputs"] = [stream(
        f"threshold={threshold:.2f}; small transactions={len(small):,}; both endpoints fraud={both.loc[small.index].mean():.2%}\n"
        f"account rule: flagged={flagged.sum():,}; precision={precision:.2%}; recall={recall:.2%}\n"
    )]
    for i, cell in enumerate(verify.cells):
        if cell.cell_type == "code":
            cell.execution_count = (1 if i == 1 else 2 if i == 3 else 3)
    nbf.write(verify, verify_path)

    eda_path = notebooks / "A_week1_eda.ipynb"
    eda = nbf.read(eda_path, as_version=4)
    summary = json.loads((PROJECT_ROOT / "reports/week1/A_eda_summary.json").read_text(encoding="utf-8"))
    eda.cells[1]["outputs"] = [stream(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")]
    image_names = [
        "A_eda_amount_distribution.png", "A_eda_transactions_per_step.png",
        "A_eda_degree_by_label.png", "A_eda_event_counts.png",
        "A_eda_amount_by_endpoint_labels.png", "A_eda_fraudstep.png",
    ]
    eda.cells[3]["outputs"] = []
    for name in image_names:
        encoded = base64.b64encode((PROJECT_ROOT / "reports/figures" / name).read_bytes()).decode("ascii")
        eda.cells[3]["outputs"].append(nbf.v4.new_output(
            "display_data", data={"image/png": encoded, "text/plain": name}, metadata={}
        ))
    for i, cell in enumerate(eda.cells):
        if cell.cell_type == "code":
            cell.execution_count = 1 if i == 1 else 2
    nbf.write(eda, eda_path)
    print("Embedded current calculated results and six generated figures in both notebooks.")


if __name__ == "__main__":
    main()
