# SPDX-License-Identifier: GPL-3.0-only
# -*- coding: utf-8 -*-
"""Historical k=2..6 feature sensitivity; five seeds, fixed RF; CSV output only."""
import importlib.util, sys, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

from rapeseed_ensemble import pipeline as ri
from rapeseed_ensemble.paths import output_path
OUT = str(output_path('feature_sensitivity'))
os.makedirs(OUT, exist_ok=True)

mod = ri.mod
mod.GL_CONFIG['min_vars'] = 1   # 让 k 从 2 起真正生效（否则 max(2,3)=3，没有真 k=2）

KS          = [2,3,4,5,6]
SEEDS_SWEEP = [42,0,1,2,3]      # Actual five-seed sensitivity setting
RF_FIXED    = {'n_estimators':500,'max_features':'sqrt','min_samples_leaf':3}
GROUPS      = ['无灾害','仅冻害','仅涝渍','干旱+冻害','干旱+冻害+涝渍']

data = mod.load_data(mod.FILE_PATH)
log=[]
def w(s):
    print(s,flush=True); log.append(str(s)); open(os.path.join(OUT,'_ksweep_log.txt'),'w',encoding='utf-8').write('\n'.join(log))

w(f'改进版 k 扫描 | device={ri.device} | seeds={SEEDS_SWEEP} | ks={KS}')
rows=[]
for k in KS:
    w(f'\n############ k={k} ############')
    mod.GL_CONFIG['target_vars']=k
    mod.set_seed(42)
    kept,gn = mod.select_features_global(data)
    keptnames=[mod.VARIABLES[i] for i in kept]
    results = ri.loyo_cv_improved(data, ri.device, kept, dict(RF_FIXED), SEEDS_SWEEP, ri.VAL_FRAC)
    dsum = mod.compute_disaster_metrics(results)
    ens,scheme,bw,wsfx,allk = mod.select_best_ensemble(results, dsum)
    for kk in ens:
        if isinstance(ens[kk],list): ens[kk]=np.array(ens[kk])
    em = mod.calc_metrics(ens['preds'],ens['acts'])
    es = mod.compute_disaster_metrics({'E':ens})['E']
    row={'k':k,'kept':';'.join(keptnames),'scheme':scheme,
         'Ensemble_R2':round(em['R2'],4),'Ensemble_RMSE':round(em['RMSE'],6),
         'TCN_R2':round(mod.calc_metrics(results['TCN']['preds'],results['TCN']['acts'])['R2'],4),
         'BiLSTM_R2':round(mod.calc_metrics(results['BiLSTM']['preds'],results['BiLSTM']['acts'])['R2'],4),
         'RF_R2':round(mod.calc_metrics(results['RandomForest']['preds'],results['RandomForest']['acts'])['R2'],4)}
    for g in GROUPS: row[f'Ens_{g}']=round(es.get(g,{}).get('R2',float('nan')),4)
    rows.append(row)
    w(f'  k={k} kept={keptnames}  集成R²={em["R2"]:.4f}  RMSE={em["RMSE"]:.5f}')

df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,'improved_k_sweep.csv'),index=False,encoding='utf-8-sig')
w('\n'+'='*60); w('改进版 k=2~6 灵敏度汇总'); w('='*60)
w(df[['k','Ensemble_R2','Ensemble_RMSE','kept']].to_string(index=False))
best=df.loc[df['Ensemble_R2'].idxmax()]
w(f"\n最高集成R²: k={int(best['k'])} ({best['Ensemble_R2']})")

# 画图
w('\n✅ 完成')
