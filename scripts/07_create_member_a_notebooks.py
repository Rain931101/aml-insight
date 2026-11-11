"""Create the two reproducible Member A Week 1 notebooks."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)
KERNEL = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
COMMON = "from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path.cwd() / 'src'))\nfrom aml_insight.data.loader import load_dataset\n"

verify = nbf.v4.new_notebook(metadata=KERNEL)
verify.cells = [
    nbf.v4.new_markdown_cell("# A 組｜Week 1 關鍵數字手動驗算\n\n直接讀主要資料集，重新計算詐欺比例、`fraudStep` 交叉表與小額交易帳戶規則；所有結果均可重跑。"),
    nbf.v4.new_code_cell(COMMON + "import pandas as pd\nds = load_dataset('20K_fanin200cycle200')\nnodes, tx = ds.nodes, ds.transactions\nprint(f'fraud accounts: {int(nodes.isFraud.sum()):,} / {len(nodes):,} = {nodes.isFraud.mean():.2%}')"),
    nbf.v4.new_markdown_cell("## `fraudStep` 與標籤交叉表\n\n`fraudStep` 只作為診斷欄位，不可作為模型特徵。"),
    nbf.v4.new_code_cell("pd.crosstab(nodes.isFraud, nodes.fraudStep.eq(-1), rownames=['isFraud'], colnames=['fraudStep == -1'])"),
    nbf.v4.new_markdown_cell("## 金額捷徑驗算\n\n以非「兩端皆詐欺帳戶」交易的最小金額作為資料中實際觀察到的分界；檢查小額交易與端點標籤的關係。"),
    nbf.v4.new_code_cell("labels = nodes.set_index('nodeid').isFraud\nboth = tx.sourceNodeId.map(labels).eq(1) & tx.targetNodeId.map(labels).eq(1)\nthreshold = tx.loc[~both, 'value'].min()\nsmall = tx.loc[tx.value < threshold]\naccounts = set(small.sourceNodeId) | set(small.targetNodeId)\nflag = nodes.nodeid.isin(accounts)\nprecision = (flag & nodes.isFraud.eq(1)).sum() / max(flag.sum(), 1)\nrecall = (flag & nodes.isFraud.eq(1)).sum() / nodes.isFraud.sum()\nprint(f'threshold={threshold:.2f}; small transactions={len(small):,}; both endpoints fraud={both.loc[small.index].mean():.2%}')\nprint(f'account rule: flagged={flag.sum():,}; precision={precision:.2%}; recall={recall:.2%}')"),
]

eda = nbf.v4.new_notebook(metadata=KERNEL)
eda.cells = [
    nbf.v4.new_markdown_cell("# A 組｜Week 1 EDA\n\n以下六張圖由主要資料集實際計算。全期間依標籤分組圖只作描述，不代表時間切分模型結果。"),
    nbf.v4.new_code_cell(COMMON + "from IPython.display import Image, display\nimport runpy\nrunpy.run_path('scripts/04_eda.py', run_name='__main__')"),
    nbf.v4.new_markdown_cell("## 圖表\n\n執行下格會顯示六張由 EDA 程式重建的圖。金額捷徑與標籤關係只反映 AMLSim 模擬設定。"),
    nbf.v4.new_code_cell("fig_dir = Path('reports/figures')\nfor name in ['A_eda_amount_distribution.png', 'A_eda_transactions_per_step.png', 'A_eda_degree_by_label.png', 'A_eda_event_counts.png', 'A_eda_amount_by_endpoint_labels.png', 'A_eda_fraudstep.png']:\n    print(name)\n    display(Image(filename=str(fig_dir / name)))"),
    nbf.v4.new_markdown_cell("### 圖表解讀\n\n1. **金額分布**：觀察偏態與小額尾部，不把金額本身當成犯罪證據。\n2. **Step 交易量**：檢查時間覆蓋；step 是模擬步數。\n3. **標籤分組入度/出度**：比較不同對手帳戶數，全期間統計僅供 EDA。\n4. **帳戶事件數**：log1p 呈現長尾活動量，事件數不同於鄰居數。\n5. **交易金額與兩端標籤**：檢查合成捷徑；兩端皆詐欺不是交易真實標籤。\n6. **fraudStep**：僅用於標籤品質診斷，永久排除於特徵。"),
]

for name, notebook in [("A_week1_verify.ipynb", verify), ("A_week1_eda.ipynb", eda)]:
    nbf.write(notebook, NOTEBOOKS / name)
    print(f"wrote {name}")
