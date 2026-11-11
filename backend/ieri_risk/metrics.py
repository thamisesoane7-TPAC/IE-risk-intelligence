"""Discrimination, calibration and uncertainty; no hidden threshold fitting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, auc, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def evaluate(y, probability, threshold: float) -> dict:
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if len(y) == 0 or len(y) != len(probability) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid binary evaluation sample")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid probability output")
    predicted = probability >= threshold
    both = len(np.unique(y)) == 2
    pr_area = None
    if both:
        precision, recall, _ = precision_recall_curve(y, probability)
        pr_area = float(auc(recall, precision))
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    bins, ece = [], 0.0
    assignments = np.minimum((probability * 10).astype(int), 9)
    for index in range(10):
        left, right = index / 10, (index + 1) / 10
        mask = assignments == index
        count = int(mask.sum())
        mean_p = float(probability[mask].mean()) if count else None
        rate = float(y[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean_p - rate)
        bins.append({"left": float(left), "right": right, "count": count,
                     "mean_probability": mean_p, "observed_stress_rate": rate})
    return {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "threshold": float(threshold), "roc_auc": float(roc_auc_score(y, probability)) if both else None,
        "average_precision": float(average_precision_score(y, probability)) if both else None,
        "pr_auc_trapezoidal": pr_area,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)), "ece_10_bins": float(ece),
        "confusion_matrix": {"tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
                             "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])},
        "calibration_bins": bins,
    }


def choose_threshold(y, probability) -> float:
    # Fixed candidate grid, selected on validation only. Ties prefer the larger threshold.
    grid = np.linspace(.05, .95, 91)
    return float(max(grid, key=lambda t: (f1_score(y, probability >= t, zero_division=0), t)))


def bootstrap_intervals(y, probability, threshold: float, seed: int, repeats: int = 200) -> dict:
    y, probability = np.asarray(y), np.asarray(probability)
    rng = np.random.default_rng(seed)
    scores = {k: [] for k in ("roc_auc", "average_precision", "brier", "precision", "recall", "f1")}
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        report = evaluate(y[indices], probability[indices], threshold)
        for name in scores:
            scores[name].append(report[name])
    return {name: {"lower_95": float(np.quantile(values, .025)) if values else None,
                   "upper_95": float(np.quantile(values, .975)) if values else None,
                   "valid_resamples": len(values)} for name, values in scores.items()}


def curves(y, probability) -> dict:
    p, r, _ = precision_recall_curve(y, probability)
    result = {"precision_recall": {"precision": p.tolist(), "recall": r.tolist()}}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, probability)
        result["roc"] = {"false_positive_rate": fpr.tolist(), "true_positive_rate": tpr.tolist()}
    return result
