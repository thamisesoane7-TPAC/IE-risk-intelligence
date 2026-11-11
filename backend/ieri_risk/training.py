"""Versioned training runs. Final test is evaluated only after selection is frozen."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from .audit import shortcut_audit
from .contracts import FEATURES, NOTICE, SCHEMA_VERSION, TARGET_VERSION, feature_frame
from .dataset import generate, load_dataset
from .explain import explain
from .io import digest, json_text, package_versions, source_digest, verify_files, write_json
from .metrics import bootstrap_intervals, curves, evaluate
from .models import fit_development, make_splits


def train(dataset: Path, registry: Path, seed: int = 20261110, trials: int = 8,
          challenge_n: int = 200, bootstrap: int = 200) -> dict:
    if not 1 <= trials <= 100 or not 100 <= challenge_n <= 10000 or not 20 <= bootstrap <= 2000:
        raise ValueError("Invalid tuning, challenge or bootstrap budget")
    registry.mkdir(parents=True, exist_ok=True)
    with (registry / ".train.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _train(dataset, registry, seed, trials, challenge_n, bootstrap)


def _train(dataset, registry, seed, trials, challenge_n, bootstrap):
    data, audit, dataset_manifest = load_dataset(dataset)
    if dataset_manifest["population"] != "development":
        raise ValueError("A challenge dataset cannot be used for model development")
    x = feature_frame(data.loc[:, list(FEATURES)])
    y = data.future_stress_30d.to_numpy(dtype=int)
    ids = data.business_id.to_numpy()
    versions = package_versions()
    config = {"seed": seed, "trials": trials, "cv_folds": 3, "challenge_n": challenge_n,
              "bootstrap_resamples": bootstrap, "dataset_manifest_sha256": digest(dataset / "manifest.json"),
              "source_sha256": source_digest(), "versions": versions}
    model_id = "ieri-sim-v1-" + hashlib.sha256(json_text(config).encode()).hexdigest()[:16]
    final = registry / model_id
    if final.exists():
        manifest = json.loads((final / "manifest.json").read_text())
        verify_files(final, manifest["files"])
        write_json(registry / "CURRENT.json", {"model_id": model_id, "manifest_sha256": digest(final / "manifest.json")})
        return {"model_id": model_id, "reused": True, "path": str(final)}
    splits = make_splits(ids, y, seed)
    print("Training: grouped CV on train; independent calibration and validation.", flush=True)
    models, bases, thresholds, champion, validation, tuning = fit_development(x, y, ids, splits, seed, trials)
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    print(f"Frozen champion: {champion}; threshold: {thresholds[champion]:.3f}", flush=True)
    background = x.iloc[tr].sample(min(32, len(tr)), random_state=seed).reset_index(drop=True)
    probes = shortcut_audit(x, y, ids, splits, bases[champion], seed)
    perm = permutation_importance(models[champion], x.iloc[va], y[va], n_repeats=5,
                                 random_state=seed, scoring="neg_brier_score", n_jobs=1)
    shap_rows = x.iloc[va].iloc[:min(12, len(va))]
    shap_values = explain(models[champion], background, shap_rows, seed)
    importance = {
        "source": "validation; not the final test", "shap_rows": len(shap_rows),
        "permutation_brier_increase": {name: {"mean": float(mean), "std": float(std)}
                                       for name, mean, std in zip(FEATURES, perm.importances_mean, perm.importances_std, strict=True)},
        "mean_absolute_calibrated_shap": dict(zip(FEATURES, np.abs(shap_values.values).mean(axis=0).tolist(), strict=True)),
        "note": "Correlated accounting features share information. Permutation/SHAP perturbations need not be feasible ledgers; attribution is not causal.",
    }
    # Selection and thresholds are already frozen. No .fit() uses final test or challenge businesses.
    test_report, test_predictions = {}, {"business_id": ids[te], "future_stress_30d": y[te]}
    for name, model in models.items():
        p = model.predict_proba(x.iloc[te])[:, 1]
        test_report[name] = {
            "calibrated": evaluate(y[te], p, thresholds[name]),
            "raw": evaluate(y[te], bases[name].predict_proba(x.iloc[te])[:, 1], thresholds[name]),
            "confidence_intervals": bootstrap_intervals(y[te], p, thresholds[name], seed + 99, bootstrap),
            "curves": curves(y[te], p),
        }
        test_predictions[name] = p
    challenge_seed = seed + 1000003
    cf, cy, ca, _ = generate(challenge_n, challenge_seed, challenge=True)
    if set(cf.business_id) & set(ids):
        raise ValueError("Challenge population overlaps development businesses")
    cx = feature_frame(cf.loc[:, list(FEATURES)])
    cp = models[champion].predict_proba(cx)[:, 1]
    challenge_report = evaluate(cy.future_stress_30d, cp, thresholds[champion])
    subgroups = {}
    for column in ("sector", "future_scenario"):
        subgroups[column] = {}
        for group in sorted(audit.iloc[te][column].unique()):
            mask = audit.iloc[te][column].to_numpy() == group
            subgroups[column][group] = evaluate(y[te][mask], np.asarray(test_predictions[champion])[mask], thresholds[champion])
    report = {
        "notice": NOTICE, "champion": champion, "champion_threshold": thresholds[champion],
        "selection_rule": "Lowest calibrated validation Brier score, then higher AP, then simpler model. Never final-test performance.",
        "validation": validation, "validation_is_selection_biased": True, "tuning": tuning,
        "final_test": test_report, "final_test_subgroups": subgroups,
        "challenge": {"seed": challenge_seed, "shift": "Stronger future shocks, extra future drift/repair frequency",
                      "calibrated": challenge_report, "used_for_selection": False},
        "importance": importance, "shortcut_audit": probes,
        "metric_notes": {
            "average_precision": "Non-interpolated precision-recall summary; random baseline is prevalence.",
            "pr_auc_trapezoidal": "Trapezoidal area is also supplied and can differ from average precision.",
            "calibration": "Sigmoid fitted only on calibration businesses. A calibration procedure does not establish good calibration.",
            "intervals": "Percentile bootstrap over unique test businesses, conditional on this fitted model and simulator; not real-world uncertainty.",
        },
    }
    temp = Path(tempfile.mkdtemp(prefix=".training-", dir=registry))
    try:
        for name, model in models.items():
            joblib.dump(model, temp / f"{name}.joblib", compress=3)
        bases["xgboost"].save_model(temp / "xgboost_base.json")
        background.to_csv(temp / "background.csv", index=False, float_format="%.17g")
        partition = pd.DataFrame({"business_id": ids, "partition": ""})
        for name, indices in splits.items():
            partition.loc[indices, "partition"] = name
        partition.to_csv(temp / "splits.csv", index=False)
        pd.DataFrame(test_predictions).to_csv(temp / "test_predictions.csv", index=False, float_format="%.17g")
        pd.DataFrame({"business_id": cf.business_id, "future_stress_30d": cy.future_stress_30d,
                      "probability": cp, "future_scenario": ca.future_scenario}).to_csv(temp / "challenge_predictions.csv", index=False)
        write_json(temp / "evaluation.json", report)
        write_json(temp / "training_config.json", config)
        write_json(temp / "dataset_manifest.json", dataset_manifest)
        freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        (temp / "environment.txt").write_text(freeze.stdout)
        metadata = {
            "model_id": model_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "champion": champion, "threshold": thresholds[champion],
            "schema_version": SCHEMA_VERSION, "target_version": TARGET_VERSION,
            "features": list(FEATURES), "history_days": 90, "horizon_days": 30,
            "models": list(models), "notice": NOTICE, "config": config,
            "feature_code_sha256": {name: digest(Path(__file__).parent / name) for name in ("contracts.py", "features.py")},
            "split_counts": {name: {"n": len(ix), "positive": int(y[ix].sum())} for name, ix in splits.items()},
            "training_ranges": {name: {"min": float(x.iloc[tr][name].min()), "max": float(x.iloc[tr][name].max())} for name in FEATURES},
            "target_definition": "Next 30 days: at least 5 days with new scheduled-payment shortfall > max(R1, 5% of due), OR peak arrears above opening arrears exceeds 7 historical mean daily obligation amounts.",
            "explanation": "Deterministic permutation SHAP on the served calibrated predictor; 32 train-only background rows; 8 permutations; probability units.",
            "limitations": ["All data are fictional; no loan-default target or real-world probability validation.",
                            "One 90+30-day window per business; no out-of-time backtest.",
                            "Small synthetic test samples yield uncertain and simulator-dependent metrics.",
                            "No automatic approvals, causal recommendations, fairness certification or authenticated public serving."],
        }
        write_json(temp / "metadata.json", metadata)
        (temp / "MODEL_CARD.md").write_text(_model_card(metadata, report))
        manifest = {"model_id": model_id, "files": {p.name: digest(p) for p in sorted(temp.iterdir()) if p.is_file()}}
        write_json(temp / "manifest.json", manifest)
        # Serialization round-trip before promotion; check both predictions and SHAP reconstruction.
        restored = joblib.load(temp / f"{champion}.joblib")
        if not np.allclose(restored.predict_proba(x.iloc[va]), models[champion].predict_proba(x.iloc[va]), atol=1e-12):
            raise ValueError("Serialized model changed predictions")
        verify_files(temp, manifest["files"])
        os.rename(temp, final)
        write_json(registry / "CURRENT.json", {"model_id": model_id, "manifest_sha256": digest(final / "manifest.json")})
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {"model_id": model_id, "champion": champion, "path": str(final), "reused": False,
            "test": test_report[champion]["calibrated"], "challenge": challenge_report,
            "shortcut_alerts": probes["alerts"]}


def _model_card(metadata, report) -> str:
    lines = ["# IERI synthetic financial-stress model card", "", NOTICE, "",
             f"Model version: `{metadata['model_id']}`. Champion: `{metadata['champion']}`.", "",
             "## Target", "", metadata["target_definition"], "",
             "## Evaluation", "", report["selection_rule"], "",
             "| Model | Test ROC-AUC | Test AP | Test Brier | Test F1 |", "|---|---:|---:|---:|---:|"]
    for name, values in report["final_test"].items():
        m = values["calibrated"]
        lines.append(f"| {name} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['brier']:.4f} | {m['f1']:.4f} |")
    lines += ["", "Calibration bins, confidence intervals, curves, subgroup counts and simulator audits are in `evaluation.json`.",
              "A model is not upgraded because it performs best on this test table. The champion was selected earlier on validation.", "",
              "## Limitations", "", *[f"- {x}" for x in metadata["limitations"]], "",
              "## Shortcut audit", "", *[f"- {x}" for x in report["shortcut_audit"]["alerts"] or ["No probe crossed its alert threshold; this does not prove the absence of shortcuts."]], ""]
    return "\n".join(lines)
