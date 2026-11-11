# Feature leakage checklist

Use `uv run python scripts/05_leakage_check.py` after rebuilding the sample data.
The script writes the checked schema and cutoff to `reports/week2/leakage_check.json`.

| Check | Rule | Programmatic evidence |
|---|---|---|
| Target separation | `isFraud` is the target only; never a feature | `build_account_features(..., include_labels=False)` and feature schema assertion |
| Direct/derived label leak | `fraudStep` is excluded from feature input | `LEAKAGE_COLUMNS`; explicit forbidden-column assertion |
| Identifier separation | `nodeid` is used to align rows only | Feature selection drops the identifier before model input |
| Temporal boundary | For cutoff `T`, transaction aggregates and graph edges use `time <= T` | `max_step` filter in `build_account_features` and `build_graph` |
| Future perturbation | Alter amount and time for every row after T; earlier features and graph must remain identical | `scripts/05_leakage_check.py` checks T=89 invariance |
| Graph labels | PyG `y` is separate from `x`; `fraudStep` never appears in `x` | `to_pyg` rejects label columns and keeps `y` separate |
| EDA vs modeling | Full-period label-group plots are descriptive only | EDA report labels full-period summaries; temporal models rebuild with a cutoff |

## Remaining modeling boundary

The account label is used to stratify later data splits and as PyG `y`; this is
expected. Graph connectivity is transductive in the agreed protocol. Any future
feature that uses label-neighbor aggregates, fraudStep, or post-cutoff events
must be reviewed and added to this checklist before it is used.
