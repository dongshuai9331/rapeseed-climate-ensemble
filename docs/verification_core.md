# Core release verification — 9 October 2026

Seven tests passed: actual defaults, model shapes, month-major feature extraction,
short inner training, synthetic LOYO interface, weight normalization/clipping,
and numerical TreeSHAP export with synthetic data and no figure output.

Twenty-eight numerical fidelity checks against the full code release passed.
These cover unchanged preprocessing, variable selection, RF tuning, model
classes, neural training, LOYO, ensemble weighting/evaluation, ablation with
only drawing omitted, and retained SHAP fitting/value/grouping calculation.
The PCA, waterlogging classification and bootstrap scripts are byte-identical
to the previously verified full release.

The new monthly SHAP and contribution summaries matched the original plotting
calculations on the archived SHAP values. The result retains approximately
60% vegetation-index, 29% soil-moisture and 11% temperature attribution.
No pure plotting imports, functions or map-rendering scripts are distributed.

Full study retraining was not performed. Unit/smoke tests do not establish
independent validation of the complete scientific pipeline. Historical global
selection/tuning and pooled weighting remain unchanged; see methods.md.
The candidate requirements are not a recovered historical lockfile.

Synthetic tables are not research observations. No raw research data, prediction
cache, private local configuration or geographic boundary data are bundled.
