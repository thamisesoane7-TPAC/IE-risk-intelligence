from pathlib import Path

import pytest

from ieri_risk.dataset import build_dataset
from ieri_risk.training import train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ieri")
    data, registry = root / "data", root / "models"
    build_dataset(data, n=240, seed=909)
    result = train(data, registry, seed=77, trials=1, challenge_n=100, bootstrap=20)
    return {"root": root, "data": data, "registry": registry, "result": result,
            "version": Path(result["path"])}
