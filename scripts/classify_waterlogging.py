# SPDX-License-Identifier: GPL-3.0-only
from rapeseed_ensemble.paths import input_path, output_path
import pandas as pd
import numpy as np

df = pd.read_excel(input_path('WATERLOGGING_INPUT', 'xun_climate_with_Qw.xlsx'))
df['year_month'] = pd.to_datetime(df['year_month'])
df['year']    = df['year_month'].dt.year
df['month']   = df['year_month'].dt.month
df['xun_num'] = df['xun'].map({'上旬': 1, '中旬': 2, '下旬': 3})
df['Q_w']     = df['Q_w'].clip(lower=0)
df = df.sort_values(['ADM1_NAME','ADM2_NAME','year','month','xun_num'])

# 计算2旬滑动平均
df['Q_w_2xun_avg'] = (
    df.groupby(['ADM1_NAME','ADM2_NAME'])['Q_w']
      .transform(lambda x: x.rolling(2, min_periods=2).mean())
)

def get_qw(grp_all, city_grp, prov, city, year, month, xun):
    """取单旬Q_w"""
    try:
        return city_grp.loc[
            (city_grp['year']==year) &
            (city_grp['month']==month) &
            (city_grp['xun_num']==xun), 'Q_w'
        ].values[0]
    except:
        return np.nan

def get_avg(city_grp, year, month, xun):
    """取2旬平均Q_w"""
    try:
        return city_grp.loc[
            (city_grp['year']==year) &
            (city_grp['month']==month) &
            (city_grp['xun_num']==xun), 'Q_w_2xun_avg'
        ].values[0]
    except:
        return np.nan

def classify(q, rules):
    if pd.isna(q):
        return '无数据'
    for label, lo, hi in rules:
        if hi is None:
            if q >= lo: return label
        else:
            if lo <= q < hi: return label
    return '无灾害'

results = []

for (prov, city), city_grp in df.groupby(['ADM1_NAME','ADM2_NAME']):
    # 作物年2014-2022：播种期在上一年10月
    for crop_year in range(2014, 2023):
        prev_year = crop_year - 1
        row = {'省': prov, '市': city, '作物年': crop_year}

        # 播种期：上一年10月，取2旬平均最大值
        q_bz = max(
            get_avg(city_grp, prev_year, 10, 2) or 0,
            get_avg(city_grp, prev_year, 10, 3) or 0
        )
        row['播种期_Q'] = round(q_bz, 4)
        row['播种期_灾害等级'] = classify(q_bz, [
            ('中度', 0.9, None),
            ('轻度', 0.8, 0.9),
        ])

        # 冬前苗期：上一年11-12月，取2旬平均最大值
        q_dm = max(
            get_avg(city_grp, prev_year, 11, 2) or 0,
            get_avg(city_grp, prev_year, 11, 3) or 0,
            get_avg(city_grp, prev_year, 12, 1) or 0,
            get_avg(city_grp, prev_year, 12, 2) or 0,
            get_avg(city_grp, prev_year, 12, 3) or 0,
        )
        row['冬前苗期_Q'] = round(q_dm, 4)
        row['冬前苗期_灾害等级'] = classify(q_dm, [
            ('轻度', 0.9, None),
        ])

        # 越冬期：当年1月-2月上旬
        q_yc = max(
            get_avg(city_grp, crop_year, 1, 1) or 0,
            get_avg(city_grp, crop_year, 1, 2) or 0,
            get_avg(city_grp, crop_year, 1, 3) or 0,
            get_avg(city_grp, crop_year, 2, 1) or 0,
        )
        row['越冬期_Q'] = round(q_yc, 4)
        row['越冬期_灾害等级'] = classify(q_yc, [
            ('轻度', 0.9, None),
        ])

        # 抽苔期：当年3月中-下旬，单旬判断
        q_ct = max(
            get_qw(df, city_grp, prov, city, crop_year, 3, 2) or 0,
            get_qw(df, city_grp, prov, city, crop_year, 3, 3) or 0
        )
        row['抽苔期_Q'] = round(q_ct, 4)
        row['抽苔期_灾害等级'] = classify(q_ct, [
            ('中度', 1.2, None),
            ('轻度', 1.0, 1.2),
        ])

        # 开花期：当年3月，2旬平均
        q_kh = max(
            get_avg(city_grp, crop_year, 3, 2) or 0,
            get_avg(city_grp, crop_year, 3, 3) or 0,
        )
        row['开花期_Q'] = round(q_kh, 4)
        row['开花期_灾害等级'] = classify(q_kh, [
            ('重度', 1.4, None),
            ('中度', 1.2, 1.4),
            ('轻度', 0.9, 1.2),
        ])

        # 灌浆期：当年4月-5月上旬，2旬平均
        q_gj = max(
            get_avg(city_grp, crop_year, 4, 2) or 0,
            get_avg(city_grp, crop_year, 4, 3) or 0,
            get_avg(city_grp, crop_year, 5, 1) or 0,
        )
        row['灌浆期_Q'] = round(q_gj, 4)
        row['灌浆期_灾害等级'] = classify(q_gj, [
            ('重度', 1.3, None),
            ('中度', 1.0, 1.3),
            ('轻度', 0.8, 1.0),
        ])

        results.append(row)

result_df = pd.DataFrame(results)
result_df.to_excel(output_path() / 'rapeseed_waterlogging_fixed.xlsx', index=False)
print(result_df.head(20).to_string())
print('\n各等级统计:')
for col in ['播种期_灾害等级','冬前苗期_灾害等级','越冬期_灾害等级',
            '抽苔期_灾害等级','开花期_灾害等级','灌浆期_灾害等级']:
    print(f'\n{col}:')
    print(result_df[col].value_counts())