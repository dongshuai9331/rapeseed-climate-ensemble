# Software provenance

The archived training scripts did not save a historical environment lockfile.
The following installed-package metadata was inspected during packaging on
2026-10-08; it cannot prove what version generated the original predictions.

| Package | Installed research environments observed |
| --- | --- |
| NumPy | 1.26.4 |
| pandas | 2.2.2 and 2.3.3 |
| SciPy | 1.13.1 and 1.12.0 |
| scikit-learn | 1.5.1 and 1.3.2 |
| Matplotlib | 3.9.2 |
| group-lasso | 1.5.0 |
| PyTorch | 2.5.1+cu121 in one research environment |
| SHAP | 0.48.0 in one research environment |
| seaborn | 0.13.2 |
| openpyxl | 3.1.5 |
| pyproj | Added for geographic distance-scale calculations, 3.7.2 |

Record a fresh successful environment with `python -m pip freeze` and the
Python version after an actual reproducibility run. Do not relabel the current
candidate requirements as an exact historical environment.
These packages are open-source projects, not software from a single corporation.
Upstream identities: Python Software Foundation (python.org), NumPy (numpy.org),
pandas (pandas.pydata.org), PyTorch (pytorch.org), scikit-learn (scikit-learn.org),
Matplotlib (matplotlib.org), SciPy (scipy.org), group-lasso
(github.com/yngvem/group-lasso), SHAP (github.com/shap/shap), pyproj
(pyproj4.github.io/pyproj). Use verified historical versions in the manuscript
if an original training environment record becomes available.
