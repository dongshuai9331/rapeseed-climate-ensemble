# Inputs

Raw study data are not bundled. Requests remain subject to the authors' ability
to share and any third-party restrictions. schema.json describes the prepared
model/PCA tables. The two bundled demo workbooks are explicitly marked synthetic.
They test interfaces only and cannot reproduce the article's quantitative findings.

Training writes predictions_cache.pkl (trusted local pickle only) and
shap_values.csv, used by bootstrap/summaries. The release begins at prepared
input tables; it does not automate raw remote-sensing data retrieval or merging.
