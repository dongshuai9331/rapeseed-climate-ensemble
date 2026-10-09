# Climate-responsive rapeseed ensemble: core computation release

Research code accompanying *An interpretable climate-responsive weighted ensemble
for rapeseed yield prediction across the middle and lower reaches of the Yangtze River*.
GPL-3.0-only. Confirm authorization from all code copyright holders before publication.

This reduced release includes computation used for the study: PCA and waterlogging
classification; global Group Lasso selection and RF tuning; TCN/BiLSTM/RF training;
LOYO predictions; regime weighting and ablation; feature-count sensitivity;
TreeSHAP values and numerical summaries; paired bootstrap comparisons.
Pure plotting, geographic rendering and optional correlation/VIF diagnostics are omitted.
The publisher does not define a file-by-file minimum; this is a computation-only package,
not a guarantee that an editor will never request further code or data.

## Installation and execution

Use Python 3.11/3.12 in a fresh environment. requirements.txt is a candidate
environment, not a recovered historical training lockfile.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy config.example.json to config.local.json, supply lawful local input paths,
and run the relevant stages. Relative paths resolve against the config folder.

```sh
python run.py pca --config config.local.json
python run.py waterlogging --config config.local.json
python run.py train --config config.local.json
python run.py sensitivity --config config.local.json
python run.py bootstrap --config config.local.json
python run.py summaries --config config.local.json
python -m unittest discover -s tests -v
```

Training can be expensive. Its outputs include predictions_cache.pkl and
shap_values.csv; point the config cache/shap fields to those output files.
Only load trusted pickle files. Training starts from the prepared city-year
table, not a raw satellite retrieval/merging workflow. The supplied demo tables
are synthetic, not article data; do not use their results as study findings.

## Scope and verification

See docs/methods.md for the actual parameters and non-nested validation boundaries.
Global selection/tuning and pooled weighting are preserved, not silently changed.
See docs/verification_core.md for release checks. No full study retraining was run.
The numerical pipeline is unchanged except that optional drawing is removed;
SHAP calculation remains and exports values rather than figures.

## Publication

Public repository: https://github.com/dongshuai9331/rapeseed-climate-ensemble

Do not upload private configs, raw study data, prediction caches or the entire
local submission folder. Cite an exact commit or versioned release when using
this code; a repository URL alone is not a permanent archival identifier.
