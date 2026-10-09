# Analysis settings and validation boundaries

The release preserves the research implementation, not a newly nested pipeline.
The numerical bodies of the neural-network training and outer LOYO routine were
retained from the original implementation. Path handling and packaging changed.

| Component | Actual setting |
| --- | --- |
| Window | Nov, Dec, Jan, Feb, Mar, Apr, May; no October model input |
| Candidate variables | NDVI, EVI, NIRv, Pre, Tem, Tmmx, Tmmn, avgVpd, avgPdsi, avgPet, avgSoil, avgSrad |
| Layout | Month-major order; 12 variables x 7 months = 84 columns |
| Group Lasso | Once on full dataset, global scaling; BIC; group_reg in 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5; l1_reg=0; scale_reg=group_size; tol=1e-5; max_iter=1000; top 5 variable groups |
| Main seeds | 42, 0, 1, 2, 3; prediction averaging within each base model |
| Outer split | Leave one observed year out; source data cover 2014-2022 |
| Outer scaling | StandardScaler and target normalization fitted to remaining years |
| Early stopping | Random 15% of outer-training rows; minimum 8 validation rows; split_seed=model seed; 500 max epochs; patience=50 |
| TCN | 48 channels; kernel 3; 3 blocks/dilations 1,2,4; dropout=0.4 |
| BiLSTM | hidden=64; 2 layers; dropout=0.25 |
| Optimizer | AdamW; TCN lr=1e-4; BiLSTM lr=3e-4; weight_decay=5e-3; batch=32; MSE; clip_grad_norm=1.0 |
| Scheduler | CosineAnnealingLR; T_max=500; eta_min=1e-5 |
| RF tuning | Once on full dataset; GridSearchCV cv=5, scoring=r2; n_estimators=[300,500,800]; max_depth=[None,10,20]; min_samples_leaf=[2,3,4,5]; max_features=[sqrt,0.5,0.6] |
| Ensemble | Regime pooled out-of-fold R2; clip negative R2 to 0; normalize R2^p; p in 1,2,3 selected using pooled predictions |
| Unit conversion | Source yield and prediction t/mu x 15000 = kg/ha |
| RMSE bootstrap | 2000 paired row resamples; RandomState(42); percentile 2.5/97.5; fixed predictions and weights, not model refitting |
| Feature-count sensitivity | k=2,3,4,5,6; actual five seeds; RF fixed at 500 trees, sqrt max_features, leaf=3 |
| SHAP | Separate full-data RF fit; TreeExplainer; RF attribution, not attribution of the complete ensemble |

Group Lasso selection and RF tuning were not repeated within each outer
training fold. Regime weights and the exponent were estimated/evaluated on
pooled out-of-fold predictions. Consequently this is not an independent
outer validation of the complete selection-and-weighting pipeline. Holding
out a year for base-model fitting does not remove these evaluation dependencies.

Drought is mean monthly PDSI < -0.5; freeze is any count below -3 C;
waterlogging is any of five provided stage labels unequal to the Chinese
no-hazard label. Verify missing stage labels before supplying new inputs:
the original classifier does not equate missing data to an explicitly observed
no-hazard status. The historical upstream waterlogging script computes an
October sowing-stage column, but that column is NOT used by load_data or the
November-May yield model. Missing-value behavior is preserved, not silently repaired.

The PCA script standardizes R/Rmax, rainy_days/n_days and sunshine/astronomical
sunshine. It uses all PCA component explained-variance ratios, absolute
weighted loadings, normalization, and coefficient ranges [0.75,1], [0.75,1],
[0.5,0.75], exactly as the original code. Fitted coefficients depend on the
supplied data and numerical library; synthetic demonstrations are not evidence
for the paper's coefficients or performance.
