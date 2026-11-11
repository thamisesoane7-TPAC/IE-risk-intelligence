import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from ieri_risk.contracts import FEATURES, SCHEMA_VERSION, ScoreRequest
from ieri_risk.dataset import load_dataset
from ieri_risk.features import calculate_features
from ieri_risk.io import digest, write_json
from ieri_risk.metrics import evaluate
from ieri_risk.models import fit_development, make_splits
from ieri_risk.service import RiskService
from ieri_risk.simulation import simulate_business


def example():
    ledger, _ = simulate_business(2, 800)
    return ScoreRequest(schema_version=SCHEMA_VERSION, business_id=ledger.business_id.iloc[0],
                        observation_end=ledger.date.iloc[89], features=calculate_features(ledger.iloc[:90]))


def test_partitions_are_business_disjoint(trained):
    partitions = pd.read_csv(trained["version"] / "splits.csv")
    assert partitions.business_id.is_unique
    assert set(partitions.partition) == {"train", "calibration", "validation", "test"}
    assert len(partitions) == 240


def test_duplicate_business_refused():
    with pytest.raises(ValueError, match="Duplicate"):
        make_splits(["duplicate"] * 100, np.tile([0, 1], 50), 2)


def test_test_data_cannot_change_selection(trained):
    data, _, _ = load_dataset(trained["data"])
    x, y, ids = data[list(FEATURES)].copy(), data.future_stress_30d.to_numpy(), data.business_id.to_numpy()
    splits = make_splits(ids, y, 77)
    # Flip labels and radically change predictors only after the same split has been fixed.
    changed_y, changed_x = y.copy(), x.copy()
    changed_y[splits["test"]] = 1 - changed_y[splits["test"]]
    changed_x.loc[splits["test"], :] = 999999.0
    fitted, _, thresholds, champion, validation, tuning = fit_development(changed_x, changed_y, ids, splits, 77, trials=1)
    original = json.loads((trained["version"] / "evaluation.json").read_text())
    assert champion == original["champion"]
    assert thresholds[champion] == original["champion_threshold"]
    assert tuning == original["tuning"]
    assert validation == original["validation"]
    restored = joblib.load(trained["version"] / f"{champion}.joblib")
    np.testing.assert_allclose(restored.predict_proba(x.iloc[splits["validation"]]),
                               fitted[champion].predict_proba(x.iloc[splits["validation"]]), atol=1e-12)


def test_all_models_and_metrics_exist(trained):
    service = RiskService(trained["registry"])
    assert set(service.report["final_test"]) == {"prior", "logistic", "random_forest", "xgboost"}
    for values in service.report["final_test"].values():
        metric = values["calibrated"]
        assert all(0 <= metric[name] <= 1 for name in ("roc_auc", "average_precision", "pr_auc_trapezoidal", "brier", "precision", "recall", "f1"))
        assert sum(b["count"] for b in metric["calibration_bins"]) == metric["n"]
    assert (trained["version"] / "xgboost_base.json").exists()


def test_calibration_boundaries_include_every_prediction():
    p = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    metric = evaluate(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]), p, .5)
    assert sum(b["count"] for b in metric["calibration_bins"]) == len(p)


def test_single_class_diagnostics_are_explicit():
    report = evaluate(np.zeros(4, dtype=int), np.array([.1, .2, .2, .4]), .5)
    assert report["roc_auc"] is None and report["average_precision"] is None


def test_serialization_and_calibrated_shap_agree(trained):
    a, b = RiskService(trained["registry"]), RiskService(trained["registry"])
    request = example()
    assert a.score(request) == b.score(request)
    result = a.explanation(request)
    e = result["explanation"]
    assert abs(e["reconstructed_probability"] - result["synthetic_stress_probability_30d"]) < 1e-6
    assert abs(e["base_value"] + sum(x["shap_value_probability"] for x in e["contributions"]) - result["synthetic_stress_probability_30d"]) < 1e-6
    assert a.explanation(request)["explanation"] == e


def test_missing_model_never_returns_a_fake_score(tmp_path):
    with pytest.raises(FileNotFoundError):
        RiskService(tmp_path)


@pytest.mark.parametrize("kind", ["tamper", "schema", "path", "dependency"])
def test_corrupt_or_incompatible_artifacts_refused(trained, tmp_path, kind):
    registry = tmp_path / "models"
    shutil.copytree(trained["registry"], registry)
    pointer = json.loads((registry / "CURRENT.json").read_text())
    version = registry / pointer["model_id"]
    if kind == "path":
        pointer["model_id"] = "../../outside"
        write_json(registry / "CURRENT.json", pointer)
    elif kind == "tamper":
        with (version / "background.csv").open("a") as f:
            f.write("tampered")
    else:
        metadata = json.loads((version / "metadata.json").read_text())
        if kind == "schema":
            metadata["schema_version"] = "wrong"
        else:
            metadata["config"]["versions"]["scikit-learn"] = "0.0.0"
        write_json(version / "metadata.json", metadata)
        manifest = json.loads((version / "manifest.json").read_text())
        manifest["files"]["metadata.json"] = digest(version / "metadata.json")
        write_json(version / "manifest.json", manifest)
        pointer["manifest_sha256"] = digest(version / "manifest.json")
        write_json(registry / "CURRENT.json", pointer)
    with pytest.raises(ValueError):
        RiskService(registry)


def test_dataset_tampering_refused(trained, tmp_path):
    destination = tmp_path / "data"
    shutil.copytree(trained["data"], destination)
    with (destination / "features.csv").open("a") as f:
        f.write("bad")
    with pytest.raises(ValueError, match="integrity"):
        load_dataset(destination)
