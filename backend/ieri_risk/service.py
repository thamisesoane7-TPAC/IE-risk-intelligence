"""Trusted local model loading and bounded, versioned inference."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contracts import FEATURES, NOTICE, SCHEMA_VERSION, TARGET_VERSION, ScoreRequest, feature_frame
from .explain import explain
from .io import digest, json_text, package_versions, verify_files


class RiskService:
    def __init__(self, registry: Path):
        pointer = json.loads((registry / "CURRENT.json").read_text())
        model_id = pointer["model_id"]
        if not re.fullmatch(r"ieri-sim-v1-[a-f0-9]{16}", model_id):
            raise ValueError("Invalid model version")
        directory = registry / model_id
        if directory.is_symlink() or directory.resolve().parent != registry.resolve():
            raise ValueError("Model path escapes its trusted registry")
        if digest(directory / "manifest.json") != pointer["manifest_sha256"]:
            raise ValueError("Model manifest integrity check failed")
        manifest = json.loads((directory / "manifest.json").read_text())
        required = {"metadata.json", "background.csv", "evaluation.json", "prior.joblib", "logistic.joblib", "random_forest.joblib", "xgboost.joblib"}
        if manifest.get("model_id") != model_id or not required.issubset(manifest.get("files", {})):
            raise ValueError("Incomplete model artifact")
        verify_files(directory, manifest["files"])
        metadata = json.loads((directory / "metadata.json").read_text())
        if (metadata["schema_version"] != SCHEMA_VERSION or metadata["target_version"] != TARGET_VERSION
                or metadata["features"] != list(FEATURES) or metadata["model_id"] != model_id):
            raise ValueError("Model feature/target schema mismatch")
        current = package_versions()
        for name, version in metadata["config"]["versions"].items():
            if name == "python":
                if current[name].split(".")[:2] != version.split(".")[:2]:
                    raise ValueError("Python major/minor mismatch; retrain in this environment")
            elif current[name] != version:
                raise ValueError(f"Runtime dependency mismatch for {name}; restore environment or retrain")
        for name in ("contracts.py", "features.py"):
            if metadata["feature_code_sha256"].get(name) != digest(Path(__file__).parent / name):
                raise ValueError("Feature code changed since training; retrain before serving")
        champion = metadata["champion"]
        if champion not in {"prior", "logistic", "random_forest", "xgboost"}:
            raise ValueError("Unknown champion model")
        # joblib is executable serialization: the operator must trust this local registry.
        self.model = joblib.load(directory / f"{champion}.joblib")
        self.background = feature_frame(pd.read_csv(directory / "background.csv"))
        self.metadata = metadata
        self.report = json.loads((directory / "evaluation.json").read_text())
        self.lock = threading.RLock()

    def score(self, request: ScoreRequest) -> dict:
        x = feature_frame(pd.DataFrame([request.features.model_dump()]))
        with self.lock, threadpool_limits(limits=1):
            probability = float(self.model.predict_proba(x)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model produced an invalid score")
        outside = [name for name in FEATURES if not (
            self.metadata["training_ranges"][name]["min"] <= x[name].iloc[0] <= self.metadata["training_ranges"][name]["max"])]
        fingerprint = hashlib.sha256(json_text(request.model_dump(mode="json")).encode()).hexdigest()
        return {
            "business_id": request.business_id, "model_id": self.metadata["model_id"],
            "model_family": self.metadata["champion"], "schema_version": SCHEMA_VERSION,
            "target_version": TARGET_VERSION, "observation_end": request.observation_end.isoformat(),
            "outcome_window_start": (request.observation_end + timedelta(days=1)).isoformat(),
            "outcome_window_end": (request.observation_end + timedelta(days=30)).isoformat(),
            "synthetic_stress_probability_30d": probability,
            "research_flag_threshold": self.metadata["threshold"],
            "research_stress_flag": bool(probability >= self.metadata["threshold"]),
            "out_of_training_range_features": outside, "input_sha256": fingerprint,
            "notice": NOTICE,
        }

    def explanation(self, request: ScoreRequest) -> dict:
        scored = self.score(request)
        x = feature_frame(pd.DataFrame([request.features.model_dump()]))
        with self.lock, threadpool_limits(limits=1):
            values = explain(self.model, self.background, x, seed=self.metadata["config"]["seed"])
        contributions = [{"feature": name, "value": float(x[name].iloc[0]),
                          "shap_value_probability": float(value), "percentage_points": float(100 * value)}
                         for name, value in zip(FEATURES, values.values[0], strict=True)]
        return {**scored, "explanation": {
            "method": "permutation_shap", "explained_output": "served_calibrated_synthetic_stress_probability_30d",
            "base_value": float(values.base_values[0]), "contributions": sorted(contributions, key=lambda r: -abs(r["shap_value_probability"])),
            "reconstructed_probability": float(values.base_values[0] + values.values[0].sum()),
            "permutations": 8, "background_rows": len(self.background),
            "interpretation": "Positive contributions increase the model score relative to this training background. They are approximate, non-causal attributions; correlated financial identities limit interpretation.",
        }}
