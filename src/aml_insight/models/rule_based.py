"""Validation-quantile rules; scores are the fraction of triggered rules."""

import numpy as np


class RuleBased:
    def __init__(self, rules, quantile=0.95):
        if not rules or not 0 < quantile < 1:
            raise ValueError("Nonempty rules and quantile in (0,1) required")
        self.rules = rules
        self.quantile = quantile
        if any(r.get("direction") not in ("high", "low") for r in rules):
            raise ValueError("Rule direction must be high or low")
        names = [r["feature"] for r in rules]
        if len(set(names)) != len(names):
            raise ValueError("Each rule feature must be unique")
        self.thresholds = {}

    def fit(self, validation_features):
        if len(validation_features) == 0:
            raise ValueError("Validation features required")
        for rule in self.rules:
            col = validation_features[rule["feature"]]
            if not np.isfinite(col).all():
                raise ValueError("Finite rule features required")
            q = self.quantile if rule["direction"] == "high" else round(1 - self.quantile, 12)
            self.thresholds[rule["feature"]] = float(col.quantile(q))
        return self

    def predict_score(self, features):
        if not self.thresholds:
            raise ValueError("Fit rules on validation first")
        hits = []
        for rule in self.rules:
            col = features[rule["feature"]].to_numpy()
            threshold = self.thresholds[rule["feature"]]
            # Quantile tail boundary is inclusive; ties are retained and reported.
            hits.append(col >= threshold if rule["direction"] == "high" else col <= threshold)
        return np.mean(hits, axis=0)
