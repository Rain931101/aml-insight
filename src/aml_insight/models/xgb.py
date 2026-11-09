"""XGBoost fitting with class weight derived only from training labels."""

import numpy as np
from xgboost import XGBClassifier

ALLOWED_PARAMS = {
    "max_depth",
    "learning_rate",
    "n_estimators",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "gamma",
}


def fit_xgb(features, labels, params, seed):
    y = np.asarray(labels)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("XGBoost training requires both binary classes")
    if set(params) - ALLOWED_PARAMS:
        raise ValueError(
            f"Unsupported or protected XGBoost parameters: {set(params) - ALLOWED_PARAMS}"
        )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
        **params,
    )
    model.fit(features, y)
    return model
