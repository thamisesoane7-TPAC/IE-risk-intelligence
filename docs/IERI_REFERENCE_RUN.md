# IERI reference execution — 11 November 2026

This report records execution in the supplied local Python 3.12 environment. It is not a claim that the user’s Codespace, original backend tests, or earlier dataset builder has been verified.

## Completed verification

- 46 new automated tests passed. One dependency deprecation warning concerned Starlette’s HTTPX test adapter; no test failed.
- Generated 500 unique fictional businesses, each with 90 observed days and 30 future days.
- Trained and tuned logistic regression, Random Forest and XGBoost, plus a prevalence baseline.
- Held out 75 businesses for final testing; evaluated another 200 shifted synthetic businesses.
- Checked serialized artifacts, runtime dependencies and calibrated SHAP reconstruction.
- Exercised installer origin parsing, existing-file preservation, idempotence and conflict refusal.
- The original backend’s 27-test result remains a handover claim; the provided phase command reruns the actual existing suite in Codespaces.

Model ID: `ieri-sim-v1-368dc6f19e321cae`. Source SHA-256: `c56cb6b24b607c262ebe419229d628447e58db7840c1cfaba6a08d644f1562a9`.

## Frozen model comparison

Random Forest was selected on validation Brier score before final-test evaluation. The research flag threshold is 0.41.

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| prior | 0.5000 | 0.2400 | 0.2400 | 1.0000 | 0.3871 | 0.1824 |
| logistic | 0.8470 | 0.7067 | 0.5455 | 0.6667 | 0.6000 | 0.1216 |
| random_forest | 0.8177 | 0.6630 | 0.6429 | 0.5000 | 0.5625 | 0.1312 |
| xgboost | 0.8275 | 0.6629 | 0.7500 | 0.3333 | 0.4615 | 0.1239 |

Logistic regression performed better on several final-test measures. The selected model was deliberately retained; switching to the test winner would contaminate model selection. This result is recorded, not hidden.

Final-test prevalence: 18/75 = 24%. The small sample makes estimates uncertain. Calibration was fitted using only 75 separate businesses, so fitted calibration is not evidence of reliable real-world probabilities.

## Shift and shortcut findings

On 200 independent shifted synthetic businesses, Random Forest ROC-AUC was 0.7830, AP 0.7381, and Brier score 0.1468. Stress prevalence was 31.5%.

Current liquidity remained informative, but its single-feature stump CV ROC-AUC was 0.7391. No predefined aggregate audit threshold fired in this run. This does not establish the absence of shortcuts.

One shuffled-label control scored about 0.70 ROC-AUC on the small validation sample; the three controls averaged near chance. This is a reminder that individual small-sample results can be misleading. All control results are retained in `ieri_reference_evaluation.json`.

The simulation includes artificial recovery/tail-expense events. It has simplified supplier terms and no real borrower outcomes. These are synthetic stress probabilities, not verified loan-default probabilities.

## Reproduction

Use `scripts/ieri_phase.sh`. It preserves the legacy dataset builder, installs a separate environment, runs the existing and new tests, and checks the real combined FastAPI application in the Codespace.

The bundled source was tested with the ML versions pinned in `backend/requirements-ieri.txt`. Model IDs include the exact Python/package versions; another Python patch version can produce a different ID. Financial/model logic is frozen in this delivery. Full metrics, curves, confidence intervals and audit outputs accompany this file.
