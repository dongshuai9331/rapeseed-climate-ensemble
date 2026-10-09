# SPDX-License-Identifier: GPL-3.0-only
"""Export paper-related numerical summaries without drawing figures."""
import pickle
import numpy as np
import pandas as pd
from rapeseed_ensemble.paths import input_path, output_path
from rapeseed_ensemble import components as c

VARS = ['NDVI', 'EVI', 'NIRv', 'SM', 'Tmax']
MONTHS = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
ORDER = ['无灾害', '仅冻害', '仅涝渍', '干旱+冻害', '冻害+涝渍', '干旱+涝渍', '干旱+冻害+涝渍']

def main():
    out = output_path('numerical_summaries')
    frame = pd.read_csv(input_path('SHAP', 'shap_values.csv'))
    features = [column for column in frame.columns if column != 'group']
    values = frame[features].values
    mean_abs = np.abs(values).mean(axis=0)
    indices = np.argsort(mean_abs)[::-1]
    pd.DataFrame({'feature': features, 'mean_abs_SHAP': mean_abs}).iloc[indices].to_csv(
        out / 'global_shap.csv', index=False)
    top = indices[:12]
    mat = np.zeros((len(top), len(ORDER)))
    for i, group in enumerate(ORDER):
        mask = frame['group'].values == group
        if mask.sum():
            mat[:, i] = np.abs(values[mask][:, top]).mean(axis=0)
    pd.DataFrame(mat, index=[features[i] for i in top], columns=ORDER).to_csv(out / 'regime_shap_top12.csv')
    monthly = np.zeros((len(VARS), len(MONTHS)))
    for i, variable in enumerate(VARS):
        for j, month in enumerate(MONTHS):
            column = f'{variable}_{month}'
            if column in frame.columns:
                monthly[i, j] = frame[column].abs().mean()
    table = pd.DataFrame(monthly, index=VARS, columns=MONTHS)
    table.loc['Total'] = monthly.sum(axis=0)
    table.to_csv(out / 'monthly_shap.csv')
    totals = monthly.sum(axis=1)
    proportions = totals / totals.sum() * 100
    pd.DataFrame({'variable': VARS, 'sum_monthly_mean_abs_SHAP': totals,
                  'share_percent': proportions}).to_csv(out / 'variable_shap_shares.csv', index=False)

    # Existing model predictions, not new model fitting.
    with input_path('CACHE', 'predictions_cache.pkl').open('rb') as stream:
        cache = pickle.load(stream)
    results = dict(cache['results'])
    results['Ensemble'] = cache['ens']
    rows = []
    for model, result in results.items():
        years = np.asarray(result['years'])
        actual, predicted = np.asarray(result['acts']), np.asarray(result['preds'])
        for year in sorted(np.unique(years)):
            mask = years == year
            metrics = c.calc_metrics(predicted[mask], actual[mask])
            rows.append({'model': model, 'year': int(year), 'R2': metrics['R2'],
                         'RMSE_kg_ha': metrics['RMSE'] * 15000,
                         'MAE_kg_ha': metrics['MAE'] * 15000, 'n': metrics['n']})
    pd.DataFrame(rows).to_csv(out / 'yearly_metrics.csv', index=False)
    print('Numerical summaries exported; no figures generated.')

if __name__ == '__main__':
    main()
