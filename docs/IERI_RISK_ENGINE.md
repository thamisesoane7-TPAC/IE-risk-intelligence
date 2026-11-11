# IERI — quantitative risk engine, phase 1

This is a complete CPU-only research implementation of forward-looking financial
stress on fictional microenterprises. It is not a loan-default model, production
credit score, lending policy or validated real-world probability estimator.

## Install and execute in Codespaces

Download the supplied `IERI_quant_engine.zip` and drag it into the VS Code Explorer
at `/workspaces/IE-risk-intelligence`, beside `backend/`. Run:

```bash
cd /workspaces/IE-risk-intelligence
python3 -m zipfile -e IERI_quant_engine.zip /tmp/ieri-quant-install
python3 /tmp/ieri-quant-install/IERI_quant_engine/install.py --repo "$PWD"
bash scripts/ieri_phase.sh
```

The installer parses the real `git remote get-url origin` and prints its actual
GitHub owner/repository. It does not assume the username in the handover is correct.
It checks all source-file conflicts before copying, accepts an identical rerun,
and refuses to replace different existing source files. Use a fresh extraction
directory if applying another release.

The phase script creates `.venv-ieri`, installs the existing backend requirements
together with the CPU ML dependencies, runs the new and existing tests, builds
500 businesses, trains all models, checks saved artifacts, creates input examples,
and smoke-tests the actual composed backend. It stops on a real failure. Your
existing `backend/app/services/ml_dataset.py` is neither called nor modified;
its execution and test status have not been assumed.

Python 3.11 or 3.12 is required. No paid APIs, GPU or Docker are used. PyPI access is
needed for initial installation. Repeated training with identical inputs, source,
configuration and dependency versions reuses the verified immutable version.

Stop the old server with Ctrl+C in its terminal, then start the integrated API:

```bash
cd /workspaces/IE-risk-intelligence/backend
../.venv-ieri/bin/python -m uvicorn ieri_asgi:app --host 0.0.0.0 --port 8000 --reload
```

Use the Codespaces forwarded port 8000 and `/docs`. Keep the forwarded port private.
The wrapper imports your original FastAPI object, preserving its routes, database
integration, middleware and lifespan, and adds the `/risk` router. It does not
replace `app/main.py`. The original command `uvicorn app.main:app` will **not** load
the new router: use `ieri_asgi:app`.

The installer also supports `backend/main.py`. An explicitly configured alternative
can be selected with `IERI_BASE_APP=module.path:app`. The model registry defaults
to `backend/artifacts/ieri` regardless of process working directory; override it
with `IERI_MODEL_DIR=/absolute/trusted/path` if necessary.

## One-shot API exercises

Run from a second terminal while the API is running:

```bash
cd /workspaces/IE-risk-intelligence/backend
curl --fail-with-body -sS http://127.0.0.1:8000/risk/health
curl --fail-with-body -sS http://127.0.0.1:8000/risk/metadata
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/risk/score \
  -H 'Content-Type: application/json' --data-binary @risk_examples/score.json
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/risk/explain \
  -H 'Content-Type: application/json' --data-binary @risk_examples/score.json
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/risk/score-history \
  -H 'Content-Type: application/json' --data-binary @risk_examples/history.json
```

| Endpoint | Purpose |
|---|---|
| `GET /risk/health` | Readiness; 503 if a verified model is unavailable |
| `GET /risk/schema` | Versioned request/feature contract; available before training |
| `GET /risk/metadata` | Model ID, feature schema, split counts, threshold, assumptions |
| `GET /risk/evaluation` | Baselines, tuned candidates, final-test metrics, curves, audits |
| `GET /risk/demo?index=0` | Fictional score and 90-day history request examples |
| `POST /risk/score` | Validated aggregate snapshot inference |
| `POST /risk/score-history` | Reconciled 90-day ledger → features → inference |
| `POST /risk/explain` | Calibrated-score SHAP with verified reconstruction |

`business_id` is a caller reference. These endpoints do not look up or register a
business in your existing SQLite database. An application can pass a registered
business's verified historical ledger to `/risk/score-history`. No automatic
mapping is invented for the unseen legacy transaction schema.

The 11 names match the earlier handover, but their exact mathematical definitions
are now versioned. Do not blindly forward old aggregate feature values: recompute
from the new ledger contract or explicitly reconcile the definitions first.
Missing features, unknown fields, numeric strings, booleans, NaN/Infinity,
inconsistent cash flow and invalid date windows are rejected. Values outside
the training ranges are reported, not silently clipped. Zero-receipt histories
are outside this model's domain and require a separate analytical treatment.

## Data design and financial assumptions

Each fictional business has exactly one 90-day observation and a disjoint,
immediately following 30-day outcome. The default business cohort is 500, seed
20261110; dates are artificial. Separate deterministic RNG streams govern latent
parameters, historical operations and future events. Changing future randomness
cannot change historical features. Business IDs, sectors, future scenario names,
latent generator parameters and target fields are never predictors. No sensitive
demographic attributes are generated for scoring.

Businesses vary in receipt scale, cost ratios, cash reserves, owner drawings,
sales volatility, trend, weekly seasonality, channel mix and temporary downturns.
Future scenarios are ordinary trading, demand contraction, supplier inflation,
operational interruption, delayed collections and compound stress. Independently,
some businesses recover; some face large unanticipated expenses. These are
explicit synthetic assumptions, not measured South African event frequencies.

Accounting is cash basis: `gross_sales_zar` means collected sales receipts;
delayed collections enter receipts when received. VAS is a subset of receipts,
not extra revenue. Supplier obligations are scheduled weekly. Operating payments
have priority, each payable pool is settled FIFO, and owner drawings occur after
available payments. Unpaid balances carry forward. Suppliers continue supplying
despite arrears, and inventory/debt/tax dynamics are simplified. These assumptions
can create shortcuts and limit economic realism.

Historical daily records reconcile:

```
gross receipts = cash receipts + card receipts + mobile receipts
net cash flow = gross receipts - supplier paid - operating paid - owner drawings
closing cash = opening cash + net cash flow
closing arrears = opening arrears + obligations due - obligations paid
```

For subsequent days, opening cash equals the prior closing cash. There are no
unrecorded capital injections, loan disbursements or balance resets.

| Feature | Definition over the observed 90 days |
|---|---|
| `average_daily_sales_zar` | Total collected sales / 90 |
| `sales_trend_30d_pct` | 100 × (receipts days 61–90 − receipts days 31–60) / max(receipts days 31–60, R1) |
| `net_cash_flow_zar` | Sum of daily net cash flows, including owner drawings |
| `cash_flow_volatility_zar` | Population standard deviation of daily net cash flow, ddof=0 |
| `negative_cash_flow_day_share` | Share of daily net cash flows below −R0.005 |
| `cash_outflow_ratio` | Actual supplier, operating and owner payments / total receipts |
| `liquidity_coverage_days` | Last observed closing cash / max(mean daily supplier + operating obligations, R1) |
| `digital_payment_share` | (Card + mobile receipts) / total receipts |
| `vas_revenue_share` | VAS receipts / total receipts |
| `payment_channel_entropy` | −Σ p log(p) / log(3), using 90-day cash/card/mobile shares; zero shares omitted |
| `supplier_due_payment_ratio` | Total supplier payments / max(total supplier obligations, R1); can exceed 1 when old arrears are cleared |

The feature validator also enforces:
`net_cash_flow_zar ≈ 90 × average_daily_sales_zar × (1 − cash_outflow_ratio)`.
This intentional accounting dependence is disclosed in the attribution analysis.

## Forward-looking target

`future_stress_30d = 1` if either condition occurs during the next 30 days:

1. At least five days have a shortfall on newly scheduled obligations greater than
   max(R1, 5% of that day's obligations), after assigning actual payments to old
   arrears first; or
2. Peak future combined supplier/operating arrears, less arrears at the observation
   cutoff, exceeds seven times the historical mean daily obligations (with R1
   minimum daily denominator).

These are research target choices. They are not regulatory definitions, verified
delinquency labels or loan-default events. Outcome data live in `outcomes.csv`,
historical aggregates in `features.csv`, simulator-only details in `audit.csv`,
and the complete reproducible trace in `ledger.csv`. Checksummed manifests bind
the files; training validates IDs, schema, finite values and window boundaries.

## Development and evaluation

| Partition | Share | Permitted use |
|---|---:|---|
| Training | 55% | Fit preprocessing/models; 3-fold stratified grouped hyperparameter search |
| Calibration | 15% | Fit sigmoid calibration on frozen base estimators |
| Validation | 15% | Select model using Brier/AP; select the research classification threshold using F1; inspect shortcuts |
| Final test | 15% | Evaluate the frozen candidates and thresholds; no fitting or selection |

Integer rounding may change exact counts. There is one observation per business,
with explicit disjointness assertions across partitions and tuning folds.
Scaling is inside the logistic pipeline; there is no global preprocessing fit.
Default random search uses eight configurations per tree family and six logistic
regularization values. All models use CPU execution and fixed seeds. No test-set
early stopping, test-driven threshold optimization, or final model refit occurs.

Models: training-prevalence prior, regularized logistic regression, Random Forest,
and XGBoost. The champion is chosen by lowest calibrated validation Brier score,
then higher average precision, then simplicity. Logistic regression or the prior
can legitimately win. A tree model is not preferred for presentation purposes.
Calibration is evaluated before and after adjustment; fitting a calibrator does
not guarantee good calibration, especially with only about 75 calibration cases.

Reports contain ROC-AUC, average precision, trapezoidal PR-AUC, precision, recall,
F1, confusion matrices, Brier score, ten-bin ECE and reliability data, ROC/PR curve
coordinates, and percentile bootstrap intervals. AP and trapezoidal PR-AUC are
named separately. Single-class subgroups return null AUC/AP. Small subgroup
results are diagnostic, not fairness claims.

The champion is also evaluated on 200 independently generated businesses with
stronger shocks and changed future drift/expense frequency. Those scores do not
select or recalibrate the model. This is a synthetic distribution-shift exercise,
not external validation or an out-of-time economic backtest.

Leakage/shortcut checks include future-seed invariance, disjoint business IDs,
explicit feature allowlists, single-feature decision stumps, an ID-only control,
shuffled-label controls and an ablation excluding liquidity and supplier-payment
ratio. Alerts are retained in the model card. A historical state that predicts a
persistent future problem is not automatically leakage, but it may mean the
simulator is too simple for a strong research claim. No probe proves the absence
of all leakage. Further code or feature choices influenced by final-test results
require a newly designed untouched test cohort before making a fresh claim.

## Explainability and artifacts

Permutation SHAP explains the actual served calibrated probability. It uses a
deterministic 32-row training background and eight permutations. Base value plus
all 11 contributions must reconstruct the score within tolerance or the endpoint
fails. Contributions are in probability units and percentage points. They are
approximate attributions, not causal advice. Perturbing correlated accounting
features can create combinations that are not realizable ledgers.

Global mean absolute SHAP and validation permutation importance are included.
Permutation importance measures the increase in Brier loss; it may be negative
on a small sample. Attribution does not establish a predictor's economic validity.

Each model version is content-addressed from source, data manifest, dependencies,
configuration and seeds. Its directory contains all four serialized estimators,
native XGBoost JSON, background rows, split assignments, predictions, reports,
model card, package versions and SHA-256 checksums. Publication updates
`CURRENT.json` atomically only after artifact checks and a serialization round-trip.

Only load artifacts created by your trusted local workflow. Joblib/pickle can
execute code, and checksums detect accidental changes but do not authenticate an
attacker-controlled file. No artifact-upload or training HTTP endpoint exists.
This phase does not add authorization, tenant access control or rate limiting;
it is intended for a private development API and synthetic data.

## Reproduce individual commands

```bash
cd /workspaces/IE-risk-intelligence/backend
../.venv-ieri/bin/python -m pytest tests_risk tests -q
../.venv-ieri/bin/python -m ieri_risk build --output risk_data/experiment_2 --businesses 500 --seed 20261111
../.venv-ieri/bin/python -m ieri_risk train --data risk_data/experiment_2 --registry artifacts/ieri --seed 20261111 --trials 8
../.venv-ieri/bin/python -m ieri_risk verify
../.venv-ieri/bin/python -m ieri_risk report
```

`build` refuses a nonempty destination. Use another named path for a new dataset;
do not edit checksummed files manually. A changed dependency or feature contract
requires retraining rather than silently serving an incompatible serialized model.
Generated datasets, environments and model binaries are ignored by Git. Export the
model card and evaluation report into `docs/` if you want a chosen research run
reviewed in your repository; do not claim it represents real borrower outcomes.

Commit this completed source phase after `ieri_phase.sh` succeeds:

```bash
cd /workspaces/IE-risk-intelligence
git add .gitignore backend/ieri_risk backend/tests_risk backend/ieri_asgi.py \
  backend/requirements-ieri.txt scripts/ieri_phase.sh docs/IERI_RISK_ENGINE.md \
  docs/IERI_REFERENCE_RUN.md docs/ieri_reference_evaluation.json
git diff --cached --stat
git commit -m "feat: add reproducible IERI quantitative stress-risk engine"
```

## Primary technical references

- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)
- [XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)
- [SHAP Explainer API](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html)

Implementation assumptions and synthetic target choices are IERI design decisions,
not conclusions established by these library references.
