"""Fit/tune only on training businesses; calibrate on a disjoint partition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .metrics import choose_threshold, evaluate


def make_splits(ids, y, seed: int) -> dict[str, np.ndarray]:
    ids, y = np.asarray(ids), np.asarray(y)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate businesses cannot be split; one observation per business is required")
    if len(np.unique(y)) != 2 or np.bincount(y.astype(int)).min() < 20:
        raise ValueError("Need at least 20 businesses in each target class; increase the dataset size")
    indices = np.arange(len(y))
    development, test = train_test_split(indices, test_size=.15, stratify=y, random_state=seed)
    fit_pool, validation = train_test_split(development, test_size=15/85, stratify=y[development], random_state=seed+1)
    train, calibration = train_test_split(fit_pool, test_size=15/70, stratify=y[fit_pool], random_state=seed+2)
    splits = {"train": train, "calibration": calibration, "validation": validation, "test": test}
    validate_splits(splits, ids)
    if any(len(np.unique(y[ix])) != 2 for ix in splits.values()):
        raise ValueError("Each partition must contain both classes")
    return splits


def validate_splits(splits, ids) -> None:
    if set(splits) != {"train", "calibration", "validation", "test"}:
        raise ValueError("Missing model-development partition")
    all_indices = np.concatenate(list(splits.values()))
    if sorted(all_indices.tolist()) != list(range(len(ids))):
        raise ValueError("Partitions overlap or omit businesses")
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated businesses would contaminate evaluation")


def candidates(seed: int):
    return {
        "logistic": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
                     {"logisticregression__C": [.03, .1, .3, 1, 3, 10]}),
        "random_forest": (RandomForestClassifier(n_jobs=1, random_state=seed), {
            "n_estimators": [120, 240], "max_depth": [3, 5, 8],
            "min_samples_leaf": [4, 8, 12], "max_features": [.7, 1.0],
        }),
        "xgboost": (XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                   tree_method="hist", device="cpu", n_jobs=1, random_state=seed), {
            "n_estimators": [100, 180, 260], "max_depth": [2, 3, 4],
            "learning_rate": [.03, .07, .12], "min_child_weight": [3, 8, 12],
            "subsample": [.75, 1.0], "colsample_bytree": [.75, 1.0], "reg_lambda": [3, 10, 30],
        }),
    }


def fit_development(x: pd.DataFrame, y: np.ndarray, ids: np.ndarray, splits: dict,
                    seed: int, trials: int = 8, folds: int = 3):
    validate_splits(splits, ids)
    train, calibration, validation = [splits[k] for k in ("train", "calibration", "validation")]
    if np.bincount(y[train]).min() < folds:
        raise ValueError("Insufficient examples per class for grouped cross-validation")
    cv = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
        x.iloc[train], y[train], groups=ids[train]))
    for a, b in cv:
        if set(ids[train][a]) & set(ids[train][b]):
            raise ValueError("Business leakage within tuning folds")
    fitted, bases, development_report, tuning = {}, {}, {}, {}
    dummy = DummyClassifier(strategy="prior").fit(x.iloc[train], y[train])
    bases["prior"], fitted["prior"] = dummy, dummy
    for name, (estimator, space) in candidates(seed).items():
        search = RandomizedSearchCV(estimator, space, n_iter=min(trials, 6) if name == "logistic" else trials,
                                    scoring="average_precision", cv=cv, random_state=seed,
                                    n_jobs=1, error_score="raise", refit=True)
        search.fit(x.iloc[train], y[train])
        base = search.best_estimator_
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        calibrated.fit(x.iloc[calibration], y[calibration])
        bases[name], fitted[name] = base, calibrated
        tuning[name] = {
            "best_params": search.best_params_, "best_cv_average_precision": float(search.best_score_),
            "candidates": [{"params": p, "mean_cv_average_precision": float(m), "std_cv_average_precision": float(s)}
                           for p, m, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                                              search.cv_results_["std_test_score"], strict=True)],
        }
    thresholds = {}
    for name, model in fitted.items():
        p = model.predict_proba(x.iloc[validation])[:, 1]
        thresholds[name] = choose_threshold(y[validation], p)
        development_report[name] = {
            "raw": evaluate(y[validation], bases[name].predict_proba(x.iloc[validation])[:, 1], thresholds[name]),
            "calibrated": evaluate(y[validation], p, thresholds[name]),
        }
    # Frozen before accessing final-test predictions: proper probability score, then ranking, then simplicity.
    priority = {"prior": 0, "logistic": 1, "random_forest": 2, "xgboost": 3}
    champion = min(fitted, key=lambda name: (development_report[name]["calibrated"]["brier"],
                                           -development_report[name]["calibrated"]["average_precision"], priority[name]))
    return fitted, bases, thresholds, champion, development_report, tuning
