"""SHAP values for the *served calibrated score*, not an uncalibrated surrogate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .contracts import FEATURES


def explain(model, background: pd.DataFrame, rows: pd.DataFrame, seed: int = 17, permutations: int = 8):
    def predict(values):
        return model.predict_proba(pd.DataFrame(values, columns=FEATURES))[:, 1]

    masker = shap.maskers.Independent(background.loc[:, list(FEATURES)], max_samples=32)
    explainer = shap.Explainer(predict, masker, algorithm="permutation", seed=seed, feature_names=list(FEATURES))
    values = explainer(rows.loc[:, list(FEATURES)], max_evals=(2 * len(FEATURES) + 1) * permutations, silent=True)
    prediction = predict(rows.to_numpy())
    reconstructed = np.asarray(values.base_values) + values.values.sum(axis=1)
    if not np.allclose(prediction, reconstructed, atol=1e-6, rtol=1e-6):
        raise ValueError("SHAP additivity check failed; explanation will not be served")
    return values
