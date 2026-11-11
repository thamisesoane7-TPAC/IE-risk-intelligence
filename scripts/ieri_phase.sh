#!/usr/bin/env bash
set -Eeuo pipefail
trap 'printf "IERI phase stopped at line %s. Fix the reported error and rerun this command.\n" "$LINENO" >&2' ERR

ieri_repo="$(git rev-parse --show-toplevel)"
cd "$ieri_repo"

# Keep dependency changes isolated from the currently running backend environment.
if [[ ! -x .venv-ieri/bin/python ]]; then
  ieri_python=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info[:2] not in ((3, 11), (3, 12)))'; then
      ieri_python="$candidate"
      break
    fi
  done
  if [[ -z "$ieri_python" ]]; then
    printf 'Use a Codespace with Python 3.11 or 3.12 for the pinned CPU ML wheels.\n' >&2
    exit 1
  fi
  "$ieri_python" -m venv .venv-ieri
fi

ieri_python="$ieri_repo/.venv-ieri/bin/python"
ieri_requirements=(-r "$ieri_repo/backend/requirements-ieri.txt")
if [[ -f backend/requirements.txt ]]; then
  ieri_requirements+=(-r "$ieri_repo/backend/requirements.txt")
elif [[ -f requirements.txt ]]; then
  ieri_requirements+=(-r "$ieri_repo/requirements.txt")
else
  ieri_requirements+=("SQLAlchemy>=2,<3")
fi
"$ieri_python" -m pip install "${ieri_requirements[@]}"
"$ieri_python" -m pip check

cd "$ieri_repo/backend"
ieri_tests=(tests_risk)
if [[ -d tests ]]; then
  ieri_tests+=(tests)
fi
# One cohesive test run, including the existing backend suite when present.
"$ieri_python" -m pytest "${ieri_tests[@]}" -q

if [[ ! -f risk_data/ieri_v1_500/manifest.json ]]; then
  "$ieri_python" -m ieri_risk build --output risk_data/ieri_v1_500 --businesses 500 --seed 20261110
fi
"$ieri_python" -m ieri_risk train --data risk_data/ieri_v1_500 --registry artifacts/ieri --seed 20261110 --trials 8
"$ieri_python" -m ieri_risk verify --registry artifacts/ieri
"$ieri_python" -m ieri_risk report --registry artifacts/ieri
"$ieri_python" -m ieri_risk example --output risk_examples/score.json
"$ieri_python" - <<'PY'
from fastapi.testclient import TestClient
from ieri_asgi import app
import json
from pathlib import Path

# Exercise the real composed backend, not a separate demonstration application.
with TestClient(app) as client:
    request = json.loads(Path('risk_examples/score.json').read_text())
    for path in ('/risk/health', '/risk/metadata', '/risk/evaluation'):
        result = client.get(path)
        assert result.status_code == 200, (path, result.status_code, result.text)
    result = client.post('/risk/score', json=request)
    assert result.status_code == 200, result.text
    explanation = client.post('/risk/explain', json=request)
    assert explanation.status_code == 200, explanation.text
    assert abs(explanation.json()['explanation']['reconstructed_probability'] - result.json()['synthetic_stress_probability_30d']) < 1e-6
print('Integrated backend, model loading, scoring and SHAP smoke checks passed.')
PY
printf '\nCompleted. Start the API with:\ncd "%s/backend"\n../.venv-ieri/bin/python -m uvicorn ieri_asgi:app --host 0.0.0.0 --port 8000 --reload\n' "$ieri_repo"
