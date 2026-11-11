"""Stable hashes and atomic local publication. Checksums are not signatures."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".ieri-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json_text(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def package_versions() -> dict[str, str]:
    import xgboost
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "shap", "joblib", "numba", "llvmlite",
    )}
    versions["xgboost"] = xgboost.__version__
    versions["python"] = platform.python_version()
    return versions


def source_digest() -> str:
    h = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        h.update(file.name.encode())
        h.update(file.read_bytes())
    return h.hexdigest()


def verify_files(folder: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = folder / name
        if path.is_symlink() or path.resolve().parent != folder.resolve():
            raise ValueError("Artifact file must be a direct, regular child of its version directory")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Artifact integrity check failed: {name}")
