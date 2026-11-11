"""Development-only shortcut probes, with findings rather than inflated assurances."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from .contracts import FEATURES


def shortcut_audit(x, y, ids, splits, base, seed: int) -> dict:
    tr, va = splits["train"], splits["validation"]
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(x.iloc[tr], y[tr], ids[tr]))
    stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=10, random_state=seed)
    single = {}
    for name in FEATURES:
        p = cross_val_predict(stump, x.iloc[tr][[name]], y[tr], cv=cv, method="predict_proba")[:, 1]
        single[name] = float(roc_auc_score(y[tr], p))
    # Numeric row index is explicitly excluded from the fitted production predictor.
    id_score = cross_val_predict(DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=seed),
                                 np.asarray(tr).reshape(-1, 1), y[tr], cv=cv, method="predict_proba")[:, 1]
    shuffled = []
    for offset in range(3):
        shuffled_y = np.random.default_rng(seed + 100 + offset).permutation(y[tr])
        model = clone(base).fit(x.iloc[tr], shuffled_y)
        shuffled.append(float(roc_auc_score(y[va], model.predict_proba(x.iloc[va])[:, 1])))
    alerts = []
    for name, score in single.items():
        if score >= .90:
            alerts.append(f"{name}: a one-feature stump has CV ROC-AUC {score:.3f}; investigate simulation persistence")
    id_auc = float(roc_auc_score(y[tr], id_score))
    if id_auc >= .65 or id_auc <= .35:
        alerts.append("Row-index-only control departed from chance; inspect generator ordering")
    if np.mean(shuffled) >= .65 or np.mean(shuffled) <= .35:
        alerts.append("Shuffled-label control departed from chance; investigate or increase sample size")
    kept = [name for name in FEATURES if name not in {"liquidity_coverage_days", "supplier_due_payment_ratio"}]
    ablated = clone(base).fit(x.iloc[tr][kept], y[tr])
    ablated_auc = float(roc_auc_score(y[va], ablated.predict_proba(x.iloc[va][kept])[:, 1]))
    full_auc = float(roc_auc_score(y[va], base.predict_proba(x.iloc[va])[:, 1]))
    return {
        "uses_final_test": False, "single_feature_stump_cv_roc_auc": single,
        "row_index_only_cv_roc_auc": id_auc, "shuffled_training_label_validation_roc_auc": shuffled,
        "liquidity_and_supplier_ratio_ablation": {
            "full_base_model_validation_roc_auc": full_auc,
            "without_liquidity_or_supplier_payment_ratio_validation_roc_auc": ablated_auc,
            "note": "A decline reveals dependence on near-target financial state; it is not by itself evidence of leakage.",
        },
        "alerts": alerts,
        "interpretation": "Small-sample probes cannot prove the absence of leakage or simulation shortcuts.",
        "structural_limits": [
            "Persistent financial health and current liquidity can make future stress predictable by construction.",
            "Weekly supplier terms, unlimited continued supply and FIFO payment priority are simplified assumptions.",
            "Channel mix and VAS share are independent nuisance features; apparent importance can be sampling noise.",
            "No demographics, sector, business ID, future scenario or future ledger fields enter the predictor.",
            "Shifted synthetic evaluation is not external, temporal or real-world validation.",
        ],
    }
