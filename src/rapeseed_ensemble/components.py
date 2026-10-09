# SPDX-License-Identifier: GPL-3.0-only
from .paths import input_path, output_path
"""Scientific components used by pipeline.py; see docs/methods.md."""
import os, math, random, warnings


warnings.filterwarnings('ignore')


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import DataLoader, TensorDataset


from sklearn.preprocessing import StandardScaler


from sklearn.ensemble import RandomForestRegressor


from sklearn.linear_model import LassoCV, Ridge


from sklearn.metrics import r2_score


try:
    from group_lasso import GroupLasso
    GROUP_LASSO_AVAILABLE = True
except ImportError:
    GROUP_LASSO_AVAILABLE = False
    print("⚠️  group-lasso 未安装，将回退到 Ridge 近似方法")


try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  shap 未安装，SHAP分析将跳过。请 pip install shap")




SEED       = 42


RESULT_DIR = str(output_path())   # 最终 k=5 配置


FILE_PATH = str(input_path('MODEL_DATA', 'mhn_data_merged3.xlsx'))


MONTHS    = ['nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'May']


VARIABLES = ['NDVI', 'EVI', 'NIRv', 'Pre', 'Tem', 'Tmmx', 'Tmmn',
             'avgVpd', 'avgPdsi', 'avgPet', 'avgSoil', 'avgSrad']


VAR_DISPLAY = {
    'NDVI'   : 'NDVI',
    'EVI'    : 'EVI',
    'NIRv'   : 'NIRv',
    'Pre'    : 'Pre',
    'Tem'    : 'Tmean',
    'Tmmx'   : 'Tmax',
    'Tmmn'   : 'Tmin',
    'avgVpd' : 'VPD',
    'avgPdsi': 'PDSI',
    'avgPet' : 'ET',
    'avgSoil': 'SM',
    'avgSrad': 'Rad',
}


FEAT_COLS   = [f'{v}_{m}' for m in MONTHS for v in VARIABLES]   # 12变量×7月 = 84列


MONTH_NAMES = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']


GL_CONFIG = {
    # group_reg 候选值列表，程序自动扫描选最优
    # 值越大 → 稀疏程度越高 → 剔除变量越多
    'group_reg_candidates': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],

    # 选择 group_reg 的标准
    # 'bic' : 贝叶斯信息准则（推荐，平衡稀疏性与拟合度）
    # 'cv'  : 5折交叉验证 MSE 最小
    'select_criterion': 'bic',

    # ── ★ 新增参数 ──────────────────────────────────────────
    # target_vars: 目标保留变量数
    #   None → 完全自动（BIC/CV决定，min_vars保底）
    #   整数 → 强制选 top-k 变量（按 Group Lasso 组范数降序）
    #          k 的合理范围：
    #           4 → 激进压缩（保留最核心特征，样本/特征比↑）
    #           5~7 → 均衡（推荐初始尝试范围）
    #           None → 保守（BIC 可能保留 6~9 个）
    #
    # 数据分析支撑（基于 mhn_data_merged3.xlsx）：
    #   - Tem/Tmmx/Tmmn 两两相关度 r>0.85 → 三选一
    #   - Pre/avgSoil 相关度 r=0.81 → 二选一
    #   - 因此 10 个变量中至少有 3 个高度冗余
    #   - 样本/特征比：
    #       10变量70特征 → 6.9:1（偏低，有过拟合风险）
    #       4变量28特征  → 17.2:1（理想范围）
    #       5变量35特征  → 13.7:1（较好）
    #       6变量42特征  → 11.5:1（可接受）
    #
    # 建议：先用 None 跑一次看 BIC 自动选几个，再决定是否设 4~6
    'target_vars': 5,   # ★ 最终选定 k=5 {NDVI,EVI,NIRv,Tmax,SM}
    # ────────────────────────────────────────────────────────

    # 最少保留的变量数（target_vars=None 时的保底；
    # target_vars 不为 None 时，target_vars 本身就是 k，min_vars 仅作下界校验）
    'min_vars': 1,   # ★ 让 target_vars 被完全遵守（不再顶到3）

    # 最多迭代次数（group-lasso 库内部优化）
    'max_iter': 1000,

    # 回退方案（group-lasso 未安装时使用 Ridge 近似）
    'ridge_alpha' : 1.0,
    'threshold'   : 0.05,
}


PDSI_COLS     = [f'avgPdsi_{m}' for m in MONTHS]


FREEZE_COLS   = ['below_minus3_nov','below_minus3_dec','below_minus3_jan',
                 'below_minus3_feb','below_minus3_mar','below_minus3_apr',
                 'below_minus3_may']


WATERLOG_COLS = ['冬前苗期_灾害等级','越冬期_灾害等级','抽苔期_灾害等级',
                 '开花期_灾害等级','灌浆期_灾害等级']


GROUP_LABELS = {
    '全部数据'       : 'All Data',
    '无灾害'         : 'No Disaster',
    '仅干旱'         : 'Drought Only',
    '仅冻害'         : 'Freeze Only',
    '仅涝渍'         : 'Waterlog Only',
    '干旱+冻害'      : 'Drought + Freeze',
    '冻害+涝渍'      : 'Freeze + Waterlog',
    '干旱+涝渍'      : 'Drought + Waterlog',
    '干旱+冻害+涝渍' : 'All Three',
}


os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(s=SEED):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


set_seed()


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_excel(path)
    missing = [c for c in FEAT_COLS if c not in data.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")

    data = data.copy()
    data['is_drought']  = (data[PDSI_COLS].mean(axis=1) < -0.5).astype(int)
    data['is_freeze']   = (data[FREEZE_COLS].sum(axis=1) > 0).astype(int)
    data['is_waterlog'] = (
        data[WATERLOG_COLS].apply(lambda c: c != '无灾害').any(axis=1)
    ).astype(int)

    print(f"数据加载成功: {len(data)} 条 | "
          f"{data['location'].nunique()} 城市 | "
          f"年份 {sorted(data['Year'].unique())}")
    print(f"灾害统计  干旱:{data['is_drought'].sum()}  "
          f"冻害:{data['is_freeze'].sum()}  "
          f"涝渍:{data['is_waterlog'].sum()}")
    return data


def select_features_global(data: pd.DataFrame) -> tuple:
    """Run Group Lasso once on the full dataset to define a common variable set.
    This is not nested feature selection within the outer LOYO folds.
    """
    print("\n" + "=" * 60)
    print("★ 全局 Group Lasso 特征筛选（全量数据，执行一次）")
    print("=" * 60)

    scaler = StandardScaler()
    X_flat = scaler.fit_transform(data[FEAT_COLS].values)
    y      = data['yield'].values.astype(np.float32)

    kept_vars, group_norms = group_lasso_select(
        X_flat, y,
        n_vars=len(VARIABLES), n_months=len(MONTHS),
        cfg=GL_CONFIG
    )

    kept_names = [VARIABLES[i] for i in kept_vars]
    dropped    = [VARIABLES[i] for i in range(len(VARIABLES))
                  if i not in kept_vars]

    print(f"\n✓ 全局筛选结果：保留 {len(kept_vars)}/{len(VARIABLES)} 个变量")
    print(f"  保留: {kept_names}")
    print(f"  剔除: {dropped}")
    print(f"  后续所有 LOYO-CV 折均使用以上变量")
    print(f"  特征形状: (N, {len(MONTHS)} 个月, {len(kept_vars)} 个变量)")
    print("=" * 60 + "\n")

    return kept_vars, group_norms


def _group_lasso_bic(X: np.ndarray,
                     y: np.ndarray,
                     groups: np.ndarray,
                     reg: float,
                     max_iter: int) -> tuple:
    """
    用指定 group_reg 拟合 Group Lasso，返回 (BIC分数, 系数向量)。
    BIC = n·ln(RSS/n) + k·ln(n)
    k = 非零系数数量；BIC 越小越好。
    """
    gl = GroupLasso(
        groups=groups,
        group_reg=reg,
        l1_reg=0,
        scale_reg='group_size',
        supress_warning=True,
        n_iter=max_iter,
        tol=1e-5,
    )
    gl.fit(X, y.reshape(-1, 1))
    residuals = y - gl.predict(X).ravel()
    rss       = np.sum(residuals ** 2)
    n         = len(y)
    k         = np.sum(np.abs(gl.coef_.ravel()) > 1e-6)
    bic = n * np.log(rss / n + 1e-10) + k * np.log(n)
    return bic, gl.coef_.ravel()


def _group_lasso_cv(X: np.ndarray,
                    y: np.ndarray,
                    groups: np.ndarray,
                    reg: float,
                    max_iter: int,
                    n_splits: int = 5) -> float:
    """
    用指定 group_reg 做 K 折 CV，返回平均 MSE。MSE 越小越好。
    """
    from sklearn.model_selection import KFold
    kf   = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    mses = []
    for tr_idx, va_idx in kf.split(X):
        gl = GroupLasso(
            groups=groups,
            group_reg=reg,
            l1_reg=0,
            scale_reg='group_size',
            supress_warning=True,
            n_iter=max_iter,
            tol=1e-5,
        )
        gl.fit(X[tr_idx], y[tr_idx].reshape(-1, 1))
        pred = gl.predict(X[va_idx]).ravel()
        mses.append(np.mean((y[va_idx] - pred) ** 2))
    return float(np.mean(mses))


def group_lasso_select(X_flat_tr: np.ndarray,
                       y_tr: np.ndarray,
                       n_vars: int = 10,
                       n_months: int = 7,
                       cfg: dict = GL_CONFIG
                       ) -> tuple:
    """
    Group Lasso 特征筛选（支持 target_vars 模式）。

    原理
    ────
    Group Lasso 在线性回归损失上添加分组惩罚：
        min  (1/2n)||y - Xβ||² + λ·Σ_g √p_g · ||β_g||₂
    λ 足够大时，某些组的系数整体收缩为零，实现整组变量剔除。

    target_vars 模式（新增）
    ────────────────────────
    当 cfg['target_vars'] 不为 None 时：
    · 仍用 BIC/CV 选出最优 λ（保证正则化强度与数据匹配）
    · 再按各组范数降序取前 target_vars 个变量（top-k选取）
    · 优点：结果数量确定，跨折一致性更强；同时 λ 选择仍有数据依据
    · 若 target_vars < min_vars，自动提升到 min_vars

    组定义
    ──────
    FEAT_COLS 列序：外层月份，内层变量
    列 j → month=j//n_vars, var=j%n_vars
    故 groups[j] = j % n_vars（同一变量7个月归为同一组）

    参数
    ────
    X_flat_tr : (N, 70)  已标准化的训练集特征
    y_tr      : (N,)     训练集产量
    n_vars    : 变量数量（10）
    n_months  : 月份数量（7）
    cfg       : GL_CONFIG 字典

    返回
    ────
    kept_vars  : 保留的变量索引列表，如 [0,1,3,5]
    group_norms: 每个变量组的系数 L2 范数（归一化），用于可视化
    """

    # ── 若 group-lasso 库未安装，回退到 Ridge 近似 ──────────
    if not GROUP_LASSO_AVAILABLE:
        return _ridge_fallback(X_flat_tr, y_tr, n_vars, n_months, cfg)

    # ── 构建组标签向量 ───────────────────────────────────────
    groups = np.array([j % n_vars for j in range(n_vars * n_months)])

    # ── 参数扫描：选最优 group_reg ───────────────────────────
    candidates  = cfg['group_reg_candidates']
    criterion   = cfg['select_criterion']
    max_iter    = cfg['max_iter']

    scores     = []
    coef_cache = {}

    print(f"    扫描 group_reg: {candidates}")
    for reg in candidates:
        if criterion == 'bic':
            score, coef = _group_lasso_bic(
                X_flat_tr, y_tr, groups, reg, max_iter)
            coef_cache[reg] = coef
        else:
            score = _group_lasso_cv(
                X_flat_tr, y_tr, groups, reg, max_iter)
            coef_cache[reg] = None
        scores.append(score)

        if coef_cache[reg] is not None:
            coef_mat_tmp    = coef_cache[reg].reshape(n_months, n_vars)
            group_norms_tmp = np.linalg.norm(coef_mat_tmp, axis=0)
            n_kept = np.sum(group_norms_tmp > 1e-6)
        else:
            n_kept = '?'
        print(f"      reg={reg:.3f}  {criterion}={score:.4f}"
              f"  GL自然选择变量数≈{n_kept}")

    best_reg = candidates[int(np.argmin(scores))]
    print(f"    ✓ 最优 group_reg = {best_reg}  "
          f"({criterion}={min(scores):.4f})")

    # ── 用最优 reg 获取系数 ──────────────────────────────────
    if criterion == 'bic' and coef_cache[best_reg] is not None:
        best_coef = coef_cache[best_reg]
    else:
        gl_best = GroupLasso(
            groups=groups,
            group_reg=best_reg,
            l1_reg=0,
            scale_reg='group_size',
            supress_warning=True,
            n_iter=max_iter,
            tol=1e-5,
        )
        gl_best.fit(X_flat_tr, y_tr.reshape(-1, 1))
        best_coef = gl_best.coef_.ravel()

    # ── 计算各组范数 ─────────────────────────────────────────
    coef_mat        = best_coef.reshape(n_months, n_vars)  # (7, 10)
    group_norms_raw = np.linalg.norm(coef_mat, axis=0)     # (10,)

    total = group_norms_raw.sum()
    group_norms_normed = (group_norms_raw / total
                          if total > 0 else group_norms_raw)

    # ── ★ 修改核心：判定保留/剔除 ────────────────────────────
    target_k = cfg.get('target_vars', None)

    if target_k is not None:
        # target_vars 模式：按组范数降序取 top-k
        # 确保 k 不低于 min_vars
        k = max(int(target_k), cfg.get('min_vars', 1))
        kept_vars = list(np.argsort(group_norms_raw)[::-1][:k])
        kept_vars.sort()
        print(f"    ✓ target_vars={k} 模式：按组范数取 top-{k} 变量")
        print(f"      各变量范数排序：")
        sorted_idx = np.argsort(group_norms_raw)[::-1]
        for rank, idx in enumerate(sorted_idx):
            marker = '★ 保留' if idx in kept_vars else '  剔除'
            print(f"      {rank+1:2d}. {VARIABLES[idx]:<10} "
                  f"norm={group_norms_raw[idx]:.6f}  {marker}")
    else:
        # 自动模式：GL 自然阈值（组范数>1e-6则保留）
        kept_vars = [i for i, n in enumerate(group_norms_raw) if n > 1e-6]

        # 保底：至少保留 min_vars 个
        if len(kept_vars) < cfg['min_vars']:
            kept_vars = list(
                np.argsort(group_norms_raw)[::-1][:cfg['min_vars']])
            kept_vars.sort()
            print(f"    ⚠️  保底机制触发，强制保留最强 {cfg['min_vars']} 个变量")

    return kept_vars, group_norms_normed


def _ridge_fallback(X_flat_tr: np.ndarray,
                    y_tr: np.ndarray,
                    n_vars: int,
                    n_months: int,
                    cfg: dict) -> tuple:
    """
    group-lasso 库未安装时的 Ridge 近似回退方案。
    同样支持 target_vars 模式。
    """
    group_norms = np.zeros(n_vars)
    for var_idx in range(n_vars):
        col_idx = [m * n_vars + var_idx for m in range(n_months)]
        X_group = X_flat_tr[:, col_idx]
        ridge   = Ridge(alpha=cfg['ridge_alpha'])
        ridge.fit(X_group, y_tr)
        group_norms[var_idx] = np.linalg.norm(ridge.coef_)

    total = group_norms.sum()
    group_norms_normed = group_norms / total if total > 0 else group_norms

    target_k = cfg.get('target_vars', None)
    if target_k is not None:
        k = max(int(target_k), cfg.get('min_vars', 1))
        kept_vars = list(np.argsort(group_norms)[::-1][:k])
        kept_vars.sort()
    else:
        kept_vars = [i for i, n in enumerate(group_norms_normed)
                     if n >= cfg['threshold']]
        if len(kept_vars) < cfg['min_vars']:
            kept_vars = list(
                np.argsort(group_norms_normed)[::-1][:cfg['min_vars']])
            kept_vars.sort()

    return kept_vars, group_norms_normed


def tune_rf_global(data: pd.DataFrame, kept_vars: list) -> dict:
    """
    在全量数据上做一次 GridSearch，确定 RF 最优超参。
    所有 LOYO-CV 折共用同一套超参（与 GL 全局筛选特征逻辑一致）。
    """
    from sklearn.model_selection import GridSearchCV

    print("\n" + "=" * 60)
    print("★ 全局 RF 超参搜索（GridSearchCV，全量数据）")
    print("=" * 60)

    scaler   = StandardScaler()
    X_flat   = scaler.fit_transform(data[FEAT_COLS].values)
    X_sel    = build_selected_features(X_flat, kept_vars,
                                        len(VARIABLES), len(MONTHS))
    X_flat_sel = X_sel.reshape(len(X_sel), -1)
    y          = data['yield'].values.astype(np.float32)

    param_grid = {
        'n_estimators'    : [300, 500, 800],
        'max_depth'       : [None, 10, 20],
        'min_samples_leaf': [2, 3, 4, 5],
        'max_features'    : ['sqrt', 0.5, 0.6],
    }

    rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    gs = GridSearchCV(rf_base, param_grid,
                      cv=5, scoring='r2',
                      n_jobs=-1, verbose=1)
    gs.fit(X_flat_sel, y)

    best = gs.best_params_
    print(f"\n✓ 最优参数: {best}")
    print(f"  CV R²={gs.best_score_:.4f}")
    print("=" * 60 + "\n")
    return best


def build_selected_features(X_flat: np.ndarray,
                             kept_vars: list,
                             n_vars: int = 10,
                             n_months: int = 7) -> np.ndarray:
    """从70维平铺特征中提取选定变量的7个月数据"""
    X_3d = X_flat.reshape(-1, n_months, n_vars)
    return X_3d[:, :, kept_vars].astype(np.float32)


class Chomp1d(nn.Module):
    """裁剪因果卷积右侧多余的 padding，保证输出长度等于输入长度。"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 基本单元：两层膨胀因果卷积 + 残差连接。
    · 因果 padding：只看当前及过去时刻，不泄漏未来信息。
    · 残差连接：若通道数不同用 1×1 卷积对齐，稳定梯度传播。
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else None)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network 主模型。

    为什么 TCN 比 BiGRU 更适合本任务（序列长度=7，样本≈480/折）
    ──────────────────────────────────────────────────────────────
    ① 感受野精准覆盖 7 步：
       kernel=3，dilation=[1,2,4]，三层后感受野 = 1+(3-1)*(1+2+4) = 15 > 7，
       完整覆盖所有月份，无需更多层。

    ② 参数量约 37k（BiGRU 约 116k）：
       参数量约为 BiGRU 的 1/3，480 条训练样本约束更充分，
       结构性地降低过拟合风险，不需要调大 dropout 或 weight_decay。

    ③ 无递归无梯度消失：
       卷积可并行，训练稳定，val loss 不会持续高于 train loss 三倍。

    ④ 更适合短序列多变量：
       RNN 的优势在长序列长距离依赖；7 步序列中，
       相邻月份的局部模式（卷积擅长捕捉）比长距离依赖更重要。

    结构：
       (B,T,C) → 转置 → 3个 TemporalBlock（dilation 1→2→4）
       → 取最后时间步 → LayerNorm + Dropout + MLP → 产量预测
    """
    def __init__(self, input_size: int = 4,
                 num_channels: int = 48,       # ★ 64→48，减小容量压制过拟合
                 kernel_size: int = 3,
                 dropout: float = 0.4):
        super().__init__()
        # dilation=[1,2,4]，感受野=15，精确覆盖7个月完整生育期
        # （kernel=3时：1+(3-1)*(1+2+4)=15 > 7，覆盖充分，无需更深）
        dilations    = [1, 2, 4]
        channel_list = [input_size] + [num_channels] * len(dilations)
        blocks = []
        for i, d in enumerate(dilations):
            blocks.append(
                TemporalBlock(channel_list[i], channel_list[i + 1],
                              kernel_size, d, dropout)
            )
        self.network = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Dropout(dropout),
            nn.Linear(num_channels, num_channels // 2),
            nn.GELU(),
            nn.Linear(num_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)  →  (B, C, T) for Conv1d
        out = self.network(x.transpose(1, 2))   # (B, num_channels, T)
        return self.head(out[:, :, -1])          # 取最后时间步


class BiLSTM(nn.Module):
    """
    双向 LSTM baseline（无 Attention）。
    池化方式：取最后一层前向与后向的最终隐状态拼接，
    即 h_n[-2]（前向末态）|| h_n[-1]（后向末态），shape=(B, hidden*2)。
    相比 Attention 池化，这是更标准的 BiLSTM 表示方式，
    计算量更低，可作为更严格的 baseline 对比。
    """
    def __init__(self, input_size=10, hidden=64,
                 num_layers=2, dropout=0.25):   # ★ 恢复2层/hidden64，dropout微调0.2→0.25
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        d_model = hidden * 2
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        # 恢复末态拼接：取最后一层前向/后向隐状态
        _, (h_n, _) = self.lstm(x)
        h_pool = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden*2)
        return self.head(h_pool)


class EarlyStopper:
    """
    带 EMA 平滑的早停器（唯一相对 v2 的改动）。
    测试集 ~60 条导致 val loss 逐 epoch 抖动，EMA 平滑趋势判断更准。
    alpha=0.3：较强平滑，patience 与 batch_size 保持 v2 原值不变。
    """
    def __init__(self, patience=25, min_delta=1e-6, ema_alpha=0.3):
        self.patience   = patience
        self.min_delta  = min_delta
        self.ema_alpha  = ema_alpha
        self.counter    = 0
        self.best_loss  = float('inf')
        self.best_state = None
        self._ema       = None

    def step(self, val_loss, model):
        if self._ema is None:
            self._ema = val_loss
        else:
            self._ema = self.ema_alpha * val_loss + (1 - self.ema_alpha) * self._ema
        if self._ema < self.best_loss - self.min_delta:
            self.best_loss  = self._ema
            self.counter    = 0
            self.best_state = {k: v.clone()
                               for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def make_loader(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(SEED))


@torch.no_grad()
def predict_nn(model, X, device, batch_size=64):
    model.eval()
    loader = make_loader(X, np.zeros(len(X)), batch_size, shuffle=False)
    preds  = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).cpu())
    return torch.cat(preds).squeeze().numpy()


def calc_metrics(preds, acts):
    if len(preds) < 2:
        return None
    rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))
    r2   = float(r2_score(acts, preds))
    mae  = float(np.mean(np.abs(preds - acts)))
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'n': len(preds)}


def _record(d, preds, acts, year, locs, drought, freeze, waterlog):
    d['preds']   .extend(preds.tolist())
    d['acts']    .extend(acts.tolist())
    d['years']   .extend([year] * len(acts))
    d['locs']    .extend(locs.tolist())
    d['drought'] .extend(drought.tolist())
    d['freeze']  .extend(freeze.tolist())
    d['waterlog'].extend(waterlog.tolist())


def get_disaster_groups(drought, freeze, waterlog):
    d = drought.astype(bool)
    f = freeze .astype(bool)
    w = waterlog.astype(bool)
    return {
        '全部数据'       : np.ones(len(d), dtype=bool),
        '无灾害'         : ~d & ~f & ~w,
        '仅干旱'         : d  & ~f & ~w,
        '仅冻害'         : f  & ~d & ~w,
        '仅涝渍'         : w  & ~d & ~f,
        '干旱+冻害'      : d  & f  & ~w,
        '冻害+涝渍'      : f  & w  & ~d,
        '干旱+涝渍'      : d  & w  & ~f,
        '干旱+冻害+涝渍' : d  & f  &  w,
    }


def compute_disaster_metrics(results):
    summary = {}
    for mname, res in results.items():
        groups = get_disaster_groups(
            res['drought'], res['freeze'], res['waterlog'])
        summary[mname] = {}
        for gname, mask in groups.items():
            if mask.sum() < 2:
                continue
            m = calc_metrics(res['preds'][mask], res['acts'][mask])
            if m:
                summary[mname][gname] = m
    return summary


MODEL_COLORS = {
    'TCN'             : '#2E86AB',
    'BiLSTM'          : '#E84855',
    'RandomForest'    : '#3BB273',
}


















def print_and_save_summary(results, disaster_summary):
    rows = []
    for mname, res in results.items():
        m = calc_metrics(res['preds'], res['acts'])
        rows.append({'Model': mname, 'Group': 'All Data (LOYO)',
                     'R2': m['R2'], 'RMSE': m['RMSE'],
                     'MAE': m['MAE'], 'n': m['n']})
    for mname in disaster_summary:
        for gname, m in disaster_summary[mname].items():
            rows.append({'Model': mname,
                         'Group': GROUP_LABELS.get(gname, gname),
                         'R2': m['R2'], 'RMSE': m['RMSE'],
                         'MAE': m['MAE'], 'n': m['n']})
    df_out = pd.DataFrame(rows)
    path   = os.path.join(RESULT_DIR, 'summary_metrics.csv')
    df_out.to_csv(path, index=False)
    print(f"\n完整指标表已保存: summary_metrics.csv")

    print("\n" + "=" * 65)
    print(f"{'模型':<22}  {'R²':>8}  {'RMSE':>10}  {'MAE':>10}  {'n':>5}")
    print("-" * 65)
    for mname, res in results.items():
        m = calc_metrics(res['preds'], res['acts'])
        print(f"{mname:<22}  {m['R2']:8.4f}  {m['RMSE']:10.6f}  "
              f"{m['MAE']:10.6f}  {m['n']:5d}")
    print("=" * 65)
    return df_out


def _get_group_name(drought: int, freeze: int, waterlog: int) -> str:
    """将单个样本的三个灾害标志映射到分组名（中文键）。"""
    d, f, w = bool(drought), bool(freeze), bool(waterlog)
    if   not d and not f and not w: return '无灾害'
    elif d  and not f and not w:    return '仅干旱'
    elif f  and not d and not w:    return '仅冻害'
    elif w  and not d and not f:    return '仅涝渍'
    elif d  and f  and not w:       return '干旱+冻害'
    elif f  and w  and not d:       return '冻害+涝渍'
    elif d  and w  and not f:       return '干旱+涝渍'
    elif d  and f  and w:           return '干旱+冻害+涝渍'
    return '全部数据'


def compute_ensemble_weights(results: dict,
                              disaster_summary: dict,
                              k: float = 1.0) -> dict:
    """
    基于 LOYO-CV 各模型在每个灾害组的 R² 计算集成权重。

    权重公式：w_i = max(R²_i, 0)^k / Σ max(R²_j, 0)^k
    k=1：直接比例；k=2/3：幂次放大模型间差距，使权重更有差异化。
    负 R² 截断为 0。
    """
    model_names = list(results.keys())

    all_groups = set()
    for mname in model_names:
        all_groups.update(disaster_summary[mname].keys())
    all_groups.add('全部数据')

    weights = {}
    for gname in all_groups:
        r2_arr = np.array([
            max(disaster_summary[mname].get(gname, {}).get('R2', 0.0), 0.0)
            for mname in model_names
        ])
        r2_pow = r2_arr ** k
        total  = r2_pow.sum()
        w = r2_pow / total if total > 0 else np.ones(len(model_names)) / len(model_names)
        weights[gname] = dict(zip(model_names, w))

    return weights


def apply_ensemble(results: dict, weights: dict) -> dict:
    """
    对每个样本按其灾害分组查找权重，加权平均三个模型的预测值。

    返回与 results['TCN'] 相同结构的 dict（可直接传入现有评估函数）。
    """
    model_names = list(results.keys())
    ref         = results[model_names[0]]
    n           = len(ref['preds'])

    ens_preds = np.zeros(n)
    for i in range(n):
        gname = _get_group_name(
            ref['drought'][i], ref['freeze'][i], ref['waterlog'][i])
        w = weights.get(gname, weights.get('全部数据', {}))
        total_w = sum(w.get(m, 0) for m in model_names)
        if total_w == 0:
            ens_preds[i] = np.mean([results[m]['preds'][i] for m in model_names])
        else:
            ens_preds[i] = sum(
                w.get(m, 0) / total_w * results[m]['preds'][i]
                for m in model_names
            )

    return {
        'preds'   : ens_preds,
        'acts'    : ref['acts'].copy(),
        'years'   : ref['years'].copy(),
        'locs'    : ref['locs'].copy(),
        'drought' : ref['drought'].copy(),
        'freeze'  : ref['freeze'].copy(),
        'waterlog': ref['waterlog'].copy(),
    }


def select_best_ensemble(results: dict,
                          disaster_summary: dict) -> tuple:
    """
    对 k=1,2,3 分别计算直接比例权重，选全局 R² 最高的方案。
    同时保留 softmax 作为对比（不参与自动选择）。

    返回：(best_ens_results, best_scheme_name,
           best_weights, softmax_weights, all_k_results)
    """
    model_names = list(results.keys())

    # ── softmax（仅用于可视化对比，不参与自动选择）─────────────
    all_groups = set()
    for mname in model_names:
        all_groups.update(disaster_summary[mname].keys())
    all_groups.add('全部数据')

    w_softmax = {}
    for gname in all_groups:
        r2_arr = np.array([
            max(disaster_summary[mname].get(gname, {}).get('R2', 0.0), 0.0)
            for mname in model_names
        ])
        w_sfx = np.exp(r2_arr) / np.exp(r2_arr).sum()
        w_softmax[gname] = dict(zip(model_names, w_sfx))

    # ── 幂次方案扫描 k=1,2,3 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("集成权重方案比较（幂次 k=1/2/3 + Softmax）")
    print("=" * 60)

    best_r2, best_k, best_weights, best_ens = -np.inf, None, None, None
    all_k_results = {}

    for k in [1, 2, 3]:
        w = compute_ensemble_weights(results, disaster_summary, k=k)
        ens = apply_ensemble(results, w)
        r2  = r2_score(ens['acts'], ens['preds'])
        rmse = np.sqrt(np.mean((ens['preds'] - ens['acts']) ** 2))
        label = f'k={k} ({"Direct" if k==1 else f"Power-{k}"})'
        print(f"  {label:<20}  R²={r2:.4f}  RMSE={rmse:.5f}")
        all_k_results[f'k={k}'] = {'weights': w, 'ens': ens, 'r2': r2}
        if r2 > best_r2:
            best_r2, best_k, best_weights, best_ens = r2, k, w, ens

    # softmax 仅打印，不参与选择
    ens_sfx = apply_ensemble(results, w_softmax)
    r2_sfx  = r2_score(ens_sfx['acts'], ens_sfx['preds'])
    rmse_sfx = np.sqrt(np.mean((ens_sfx['preds'] - ens_sfx['acts']) ** 2))
    print(f"  {'Softmax (参考)':<20}  R²={r2_sfx:.4f}  RMSE={rmse_sfx:.5f}")

    scheme_name = f'Power k={best_k}' if best_k > 1 else 'Direct (k=1)'
    print(f"\n  ✓ 自动选择：{scheme_name}  (R²={best_r2:.4f})")
    print("=" * 60)

    # 打印最优方案权重表
    print(f"\n最优方案权重表（{scheme_name}）：")
    print(f"{'分组':<22}" + "".join(f"  {m[:10]:>12}" for m in model_names))
    print("-" * 60)
    for gname in sorted(best_weights.keys()):
        row = f"{GROUP_LABELS.get(gname, gname):<22}"
        row += "".join(f"  {best_weights[gname][m]:12.4f}" for m in model_names)
        print(row)

    return best_ens, scheme_name, best_weights, w_softmax, all_k_results




def run_shap_analysis(data: pd.DataFrame,
                       results: dict,
                       kept_vars: list):
    """Fit the historical full-data RF and export raw TreeSHAP values; no figures."""
    if not SHAP_AVAILABLE:
        print("⚠️  shap 未安装，跳过SHAP分析")
        return

    print("\n" + "=" * 60)
    print("★ RF TreeSHAP 分析")
    print("=" * 60)

    # ── 准备特征矩阵 ──────────────────────────────────────────
    scaler   = StandardScaler()
    X_flat   = scaler.fit_transform(data[FEAT_COLS].values)
    X_sel    = build_selected_features(X_flat, kept_vars,
                                        len(VARIABLES), len(MONTHS))
    X_flat_sel = X_sel.reshape(len(X_sel), -1).astype(np.float32)
    y          = data['yield'].values.astype(np.float32)

    # 特征名：变量_月份，用显示名
    kept_var_names = [VARIABLES[i] for i in kept_vars]
    feat_names = [f'{VAR_DISPLAY.get(v,v)}_{m}'
                  for m in MONTH_NAMES
                  for v in kept_var_names]

    # ── 训练全量RF ───────────────────────────────────────────
    print("  训练全量RF模型...")
    rf_full = RandomForestRegressor(
        n_estimators=500, max_features='sqrt',
        min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    rf_full.fit(X_flat_sel, y)

    # ── 计算SHAP值 ───────────────────────────────────────────
    print("  计算TreeSHAP值...")
    explainer   = shap.TreeExplainer(rf_full)
    shap_values = explainer.shap_values(X_flat_sel)  # (N, n_feats)

    # 构建灾害分组标签
    data2 = data.copy()
    data2['is_drought']  = (data2[PDSI_COLS].mean(axis=1) < -0.5).astype(int)
    data2['is_freeze']   = (data2[FREEZE_COLS].sum(axis=1) > 0).astype(int)
    data2['is_waterlog'] = (
        data2[WATERLOG_COLS].apply(lambda c: c != '无灾害').any(axis=1)
    ).astype(int)

    group_labels = np.array([
        _get_group_name(int(row['is_drought']),
                        int(row['is_freeze']),
                        int(row['is_waterlog']))
        for _, row in data2.iterrows()
    ])

    df_shap = pd.DataFrame(shap_values, columns=feat_names)
    df_shap['group'] = group_labels
    df_shap.to_csv(os.path.join(RESULT_DIR, 'shap_values.csv'), index=False)
    print('SHAP values saved: shap_values.csv')
