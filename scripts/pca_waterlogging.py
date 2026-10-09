# SPDX-License-Identifier: GPL-3.0-only
from rapeseed_ensemble.paths import input_path, output_path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_excel(input_path('PCA_INPUT', 'xun_climate_merged.xlsx'))

# 构建三个指标
df['x1'] = df['prcp_mm'] / df['hist_max_prcp_mm'].replace(0, np.nan)
df['x2'] = df['rainy_days'] / df['n_days']
df['x3'] = df['sun_hours'] / df['astro_sun_hrs'].replace(0, np.nan)
df = df.dropna(subset=['x1', 'x2', 'x3'])

# 标准化
X_scaled = StandardScaler().fit_transform(df[['x1', 'x2', 'x3']])

# PCA
pca = PCA()
pca.fit(X_scaled)

variance_ratio = pca.explained_variance_ratio_
loadings = pca.components_.T  # shape: (3, 3)

print('各主成分方差贡献率:', variance_ratio.round(4))
print('累计贡献率:', variance_ratio.cumsum().round(4))
print()
print('载荷矩阵:')
print(pd.DataFrame(loadings.round(4),
      index=['x1(R/Rmax)', 'x2(DR/D)', 'x3(S/Smax)'],
      columns=['PC1', 'PC2', 'PC3']))

# 综合权重
weights = np.abs(loadings @ variance_ratio)
weights_normalized = weights / weights.sum()
print()
print(f'归一化权重: x1={weights_normalized[0]:.4f}, x2={weights_normalized[1]:.4f}, x3={weights_normalized[2]:.4f}')

# 映射到标准参考范围
def scale_to_range(w, low, high):
    return low + w * (high - low)

b1 = scale_to_range(weights_normalized[0], 0.75, 1.00)
b2 = scale_to_range(weights_normalized[1], 0.75, 1.00)
b3 = scale_to_range(weights_normalized[2], 0.50, 0.75)
print()
print(f'最终系数: b1={b1:.3f}, b2={b2:.3f}, b3={b3:.3f}')

# 计算涝渍指数并保存
df['Q_w'] = (
    b1 * df['x1'] +
    b2 * df['x2'] -
    b3 * df['x3']
)
df.to_excel(output_path() / 'xun_climate_with_Qw.xlsx', index=False)
print()
print('已保存到 xun_climate_with_Qw.xlsx')
print(df[['ADM1_NAME','ADM2_NAME','year_month','xun','Q_w']].head(10))