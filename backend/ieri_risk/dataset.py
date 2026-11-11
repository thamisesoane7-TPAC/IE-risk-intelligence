"""Separate historical features, future labels, audit metadata and the full ledger."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .contracts import FEATURES, SCHEMA_VERSION, SIMULATOR_VERSION, TARGET_VERSION, feature_frame
from .features import calculate_features
from .io import digest, write_json, verify_files
from .simulation import forward_outcome, simulate_business


def generate(n: int = 500, seed: int = 20261110, *, challenge: bool = False):
    if n < 100 or n > 100000 or seed < 0:
        raise ValueError("Use 100–100000 businesses and a nonnegative seed")
    features, outcomes, audits, ledgers = [], [], [], []
    for index in range(n):
        ledger, audit = simulate_business(index, seed, challenge=challenge)
        history, future = ledger.iloc[:90].copy(), ledger.iloc[90:].copy()
        features.append({"business_id": audit["business_id"],
                         "observation_start": history.date.iloc[0],
                         "observation_end": history.date.iloc[-1], **calculate_features(history)})
        outcomes.append(forward_outcome(history, future))
        audits.append(audit)
        ledgers.append(ledger)
    return pd.DataFrame(features), pd.DataFrame(outcomes), pd.DataFrame(audits), pd.concat(ledgers, ignore_index=True)


def build_dataset(destination: Path, n: int = 500, seed: int = 20261110, *, challenge: bool = False) -> Path:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Dataset directory is nonempty: {destination}. Use a new path.")
    destination.mkdir(parents=True, exist_ok=True)
    frames = generate(n, seed, challenge=challenge)
    names = ("features.csv", "outcomes.csv", "audit.csv", "ledger.csv")
    for name, frame in zip(names, frames, strict=True):
        frame.to_csv(destination / name, index=False, float_format="%.12g")
    write_json(destination / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "target_version": TARGET_VERSION,
        "simulator_version": SIMULATOR_VERSION, "seed": seed, "n_businesses": n,
        "population": "shifted_challenge" if challenge else "development",
        "history_days": 90, "horizon_days": 30, "features": list(FEATURES),
        "files": {name: digest(destination / name) for name in names},
    })
    load_dataset(destination)
    return destination


def load_dataset(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("target_version") != TARGET_VERSION
            or manifest.get("simulator_version") != SIMULATOR_VERSION or manifest.get("features") != list(FEATURES)
            or manifest.get("history_days") != 90 or manifest.get("horizon_days") != 30):
        raise ValueError("Dataset schema/target/simulator mismatch; do not reuse an unverified legacy builder")
    if set(manifest.get("files", {})) != {"features.csv", "outcomes.csv", "audit.csv", "ledger.csv"}:
        raise ValueError("Incomplete dataset manifest")
    verify_files(path, manifest["files"])
    features, outcomes, audit = [pd.read_csv(path / name) for name in ("features.csv", "outcomes.csv", "audit.csv")]
    if set(features.columns) != {"business_id", "observation_start", "observation_end", *FEATURES}:
        raise ValueError("Historical feature file contains unexpected columns")
    for frame in (features, outcomes, audit):
        if frame.business_id.isna().any() or frame.business_id.duplicated().any() or len(frame) != manifest["n_businesses"]:
            raise ValueError("Exactly one unique observation per business is required")
        if set(frame.business_id) != set(features.business_id):
            raise ValueError("Business identity mismatch between dataset files")
    merged = features.merge(outcomes, on="business_id", validate="one_to_one")
    if not merged.future_stress_30d.isin([0, 1]).all():
        raise ValueError("Target must be binary")
    start, end = pd.to_datetime(merged.observation_start), pd.to_datetime(merged.observation_end)
    outcome_start, outcome_end = pd.to_datetime(merged.outcome_start), pd.to_datetime(merged.outcome_end)
    if not ((end - start).dt.days.eq(89) & (outcome_start - end).dt.days.eq(1)
            & (outcome_end - outcome_start).dt.days.eq(29)).all():
        raise ValueError("Invalid or overlapping observation/outcome windows")
    feature_frame(merged.loc[:, list(FEATURES)])
    return merged, audit.set_index("business_id").loc[merged.business_id].reset_index(), manifest
