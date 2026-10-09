# SPDX-License-Identifier: GPL-3.0-only
"""Published analysis implementation; LOYO training with inner early stopping.
See docs/methods.md for non-nested selection and weighting boundaries.
"""
import importlib.util, sys, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

from .paths import input_path, output_path
OUT = str(output_path())
os.makedirs(OUT, exist_ok=True)

# ---------------- 配置 ----------------
MONTHS = ['nov','dec','jan','feb','mar','apr','May']
ALL12  = ['NDVI','EVI','NIRv','Pre','Tem','Tmmx','Tmmn','avgVpd','avgPdsi','avgPet','avgSoil','avgSrad']
SEEDS      = [42, 0, 1, 2, 3]   # 深度集成的随机种子
VAL_FRAC   = 0.15               # 训练年内部验证集比例(早停用)
TARGET_VARS= 5   # ★ 最终选定 k=5 {NDVI,EVI,NIRv,Tmax,SM}
USE_GRIDSEARCH = True           # RF 是否全局 GridSearch(只跑一次)

# ---------------- 导入原模块 ----------------
from . import components as mod

import torch, torch.nn as nn
mod.RESULT_DIR = OUT                       # 让原绘图/CSV 写到本文件夹
mod.VARIABLES  = list(ALL12)
mod.FEAT_COLS  = [f'{v}_{m}' for m in MONTHS for v in mod.VARIABLES]
mod.GL_CONFIG['target_vars'] = TARGET_VARS
mod.GL_CONFIG['min_vars'] = 1   # 让 target_vars 被完全遵守（否则 max(k,min_vars) 会顶到 3）

device = torch.device('cpu')   # ★ 强制 CPU（小模型 CPU 通常比 GPU 快）

# ===============================================================
# (a) 无泄漏训练：训练集内部再划验证子集做早停，绝不碰测试集
# ===============================================================
def train_nn_noleak(model, X_tr, y_tr, *, epochs=500, lr=1e-4, batch_size=32,
                    patience=50, device, val_frac=0.15, split_seed=0):
    n = len(X_tr)
    rng = np.random.RandomState(split_seed)
    idx = rng.permutation(n); nv = max(8, int(n*val_frac))
    va_idx, tr_idx = idx[:nv], idx[nv:]
    Xtr, ytr = X_tr[tr_idx], y_tr[tr_idx]
    Xva, yva = X_tr[va_idx], y_tr[va_idx]      # ← 验证来自训练年内部

    tr_loader = mod.make_loader(Xtr, ytr, batch_size, shuffle=True)
    va_loader = mod.make_loader(Xva, yva, batch_size, shuffle=False)
    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    stopper = mod.EarlyStopper(patience=patience)
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                vloss += criterion(model(xb.to(device)), yb.to(device)).item()
        if stopper.step(vloss/len(va_loader), model):
            break
    stopper.restore(model)
    return model

# ===============================================================
# (b) 多种子深度集成的 LOYO
# ===============================================================
def loyo_cv_improved(data, device, kept_vars, rf_params, seeds, val_frac):
    years = sorted(data['Year'].unique()); input_size = len(kept_vars)
    kept_names = [mod.VARIABLES[i] for i in kept_vars]
    print(f'\n[Improved] LOYO {len(years)}折 | 设备{device} | 种子{seeds} | 变量{kept_names}', flush=True)
    R = {m: {k: [] for k in ['preds','acts','years','locs','drought','freeze','waterlog']}
         for m in ['TCN','BiLSTM','RandomForest']}
    for fi, ty in enumerate(years):
        print(f'  Fold {fi+1}/{len(years)}  测试年={ty}', flush=True)
        trm = data['Year'].values != ty; tem = data['Year'].values == ty
        scaler = mod.StandardScaler()
        Xtr = scaler.fit_transform(data.loc[trm, mod.FEAT_COLS].values)
        Xte = scaler.transform(data.loc[tem, mod.FEAT_COLS].values)
        ytr_raw = data.loc[trm,'yield'].values.astype(np.float32)
        yte_raw = data.loc[tem,'yield'].values.astype(np.float32)
        ym, ys = ytr_raw.mean(), ytr_raw.std()
        ytr_n = (ytr_raw-ym)/ys
        Xtr_s = mod.build_selected_features(Xtr, kept_vars, len(mod.VARIABLES), len(MONTHS))
        Xte_s = mod.build_selected_features(Xte, kept_vars, len(mod.VARIABLES), len(MONTHS))
        Xtr_f = Xtr_s.reshape(len(Xtr_s),-1); Xte_f = Xte_s.reshape(len(Xte_s),-1)
        locs = data.loc[tem,'location'].values
        dr = data.loc[tem,'is_drought'].values; fz = data.loc[tem,'is_freeze'].values; wl = data.loc[tem,'is_waterlog'].values

        # ---- TCN: 多种子平均 ----
        ptcn = []
        for sd in seeds:
            mod.set_seed(sd)
            m = mod.TCN(input_size=input_size, num_channels=48, kernel_size=3, dropout=0.4).to(device)
            train_nn_noleak(m, Xtr_s, ytr_n, epochs=500, lr=1e-4, batch_size=32,
                            patience=50, device=device, val_frac=val_frac, split_seed=sd)
            ptcn.append(mod.predict_nn(m, Xte_s, device)*ys+ym)
        p_tcn = np.mean(ptcn, axis=0)

        # ---- BiLSTM: 多种子平均 ----
        plstm = []
        for sd in seeds:
            mod.set_seed(sd)
            m = mod.BiLSTM(input_size=input_size, hidden=64, num_layers=2, dropout=0.25).to(device)
            train_nn_noleak(m, Xtr_s, ytr_n, epochs=500, lr=3e-4, batch_size=32,
                            patience=50, device=device, val_frac=val_frac, split_seed=sd)
            plstm.append(mod.predict_nn(m, Xte_s, device)*ys+ym)
        p_lstm = np.mean(plstm, axis=0)

        # ---- RF: 多种子平均 ----
        prf = []
        for sd in seeds:
            rf = mod.RandomForestRegressor(**rf_params, random_state=sd, n_jobs=-1)
            rf.fit(Xtr_f, ytr_raw); prf.append(rf.predict(Xte_f))
        p_rf = np.mean(prf, axis=0)

        for nm, pr in [('TCN',p_tcn),('BiLSTM',p_lstm),('RandomForest',p_rf)]:
            mod._record(R[nm], pr, yte_raw, ty, locs, dr, fz, wl)
        print(f'    TCN R²={mod.r2_score(yte_raw,p_tcn):.4f}  '
              f'BiLSTM R²={mod.r2_score(yte_raw,p_lstm):.4f}  '
              f'RF R²={mod.r2_score(yte_raw,p_rf):.4f}', flush=True)
    for nm in R:
        for k in R[nm]: R[nm][k] = np.array(R[nm][k])
    return R

# ===============================================================
# 损失曲线（无泄漏版：训练 vs 训练年内部验证，并标早停点）
# ===============================================================


# ===============================================================
# 集成策略消融：等权 / 全局R²加权(不分灾害) / 灾害分层加权
# ===============================================================
def ensemble_ablation(results, dsum):
    mns=list(results.keys()); groups=set(['全部数据'])
    for m in mns: groups.update(dsum[m].keys())
    def r2_of(w):
        e=mod.apply_ensemble(results,w); return mod.r2_score(e['acts'],e['preds']), e
    w_eq={g:{m:1.0/len(mns) for m in mns} for g in groups}; r2_eq,_=r2_of(w_eq)
    rows=[]; best=None
    for k in [1,2,3]:
        w_s=mod.compute_ensemble_weights(results,dsum,k=k); r2_s,ens_s=r2_of(w_s)
        gw=w_s.get('全部数据'); w_g={g:dict(gw) for g in groups}; r2_g,_=r2_of(w_g)
        rows.append({'power_k':k,'Equal_avg':round(r2_eq,4),'Global_Rweight':round(r2_g,4),
                     'Disaster_stratified':round(r2_s,4),'gain_strat_minus_global':round(r2_s-r2_g,4)})
        if best is None or r2_s>best[1]: best=(k,r2_s,w_s)
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'ensemble_ablation.csv'),index=False,encoding='utf-8-sig')
    # 最优k下，分组比较 全局 vs 分层
    bk,_,w_s=best; gw=w_s.get('全部数据'); w_g={g:dict(gw) for g in groups}
    ens_s=mod.apply_ensemble(results,w_s); ens_g=mod.apply_ensemble(results,w_g)
    ss=mod.compute_disaster_metrics({'S':ens_s})['S']; gs=mod.compute_disaster_metrics({'G':ens_g})['G']
    glist=['全部数据','无灾害','仅冻害','仅涝渍','干旱+冻害','冻害+涝渍','干旱+涝渍','干旱+冻害+涝渍']
    pr=[{'group':mod.GROUP_LABELS.get(g,g),
         'Global':round(gs.get(g,{}).get('R2',float('nan')),4),
         'Stratified':round(ss.get(g,{}).get('R2',float('nan')),4)} for g in glist]
    pd.DataFrame(pr).to_csv(os.path.join(OUT,'ensemble_ablation_bygroup.csv'),index=False,encoding='utf-8-sig')
    print('\n[集成消融] 各方案整体 R²(best power k=%d):'%bk, flush=True)
    print('  等权=%.4f  全局加权=%.4f  灾害分层=%.4f  (分层-全局=%+.4f)'%(
        r2_eq, rows[bk-1]['Global_Rweight'], rows[bk-1]['Disaster_stratified'],
        rows[bk-1]['gain_strat_minus_global']), flush=True)

# ===============================================================
# 主流程
# ===============================================================
def main():
    if not mod.GROUP_LASSO_AVAILABLE:
        raise RuntimeError('Install group-lasso; the release does not substitute Ridge for Group Lasso.')
    data = mod.load_data(mod.FILE_PATH)
    mod.set_seed(42)
    kept_vars, group_norms = mod.select_features_global(data)
    rf_params = mod.tune_rf_global(data, kept_vars) if USE_GRIDSEARCH else \
                {'n_estimators':500,'max_features':'sqrt','min_samples_leaf':3}

    results = loyo_cv_improved(data, device, kept_vars, rf_params, SEEDS, VAL_FRAC)
    dsum = mod.compute_disaster_metrics(results)
    ens, scheme, bw, wsfx, allk = mod.select_best_ensemble(results, dsum)
    for k in ens:
        if isinstance(ens[k], list): ens[k] = np.array(ens[k])
    ens_sum = mod.compute_disaster_metrics({'Ensemble': ens})['Ensemble']
    em = mod.calc_metrics(ens['preds'], ens['acts'])
    print(f"\n[改进版] 集成 R²={em['R2']:.4f}  RMSE={em['RMSE']:.6f}  MAE={em['MAE']:.6f}", flush=True)

    # 汇总表
    df = mod.print_and_save_summary(results, dsum)
    erows = [{'Model':'Ensemble','Group':'All Data (LOYO)','R2':em['R2'],'RMSE':em['RMSE'],'MAE':em['MAE'],'n':em['n']}]
    for g,m in ens_sum.items():
        erows.append({'Model':'Ensemble','Group':mod.GROUP_LABELS.get(g,g),'R2':m['R2'],'RMSE':m['RMSE'],'MAE':m['MAE'],'n':m['n']})
    pd.concat([df, pd.DataFrame(erows)], ignore_index=True).to_csv(os.path.join(OUT,'summary_metrics.csv'), index=False, encoding='utf-8-sig')

    # ★ 集成策略消融：全局加权 vs 灾害分层加权
    try: ensemble_ablation(results, dsum)
    except Exception as e: print('消融跳过:', e)

    # 缓存预测，便于以后重做消融/分析而无需重训
    try:
        import pickle
        with open(os.path.join(OUT,'predictions_cache.pkl'),'wb') as f:
            pickle.dump({'results':results,'ens':ens,'dsum':dsum}, f)
        print('已缓存预测: predictions_cache.pkl', flush=True)
    except Exception as e: print('缓存跳过:', e)

    # Export numerical RF attribution; plotting omitted in the core release.
    mod.run_shap_analysis(data, results, kept_vars)
    print(f'\n✅ 改进版完成，结果在 {OUT}', flush=True)

if __name__ == '__main__':
    main()
