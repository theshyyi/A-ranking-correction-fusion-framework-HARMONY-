import os
import json
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心指标计算函数 (基于样本值)
# ==========================================
def calculate_metrics_from_samples(p, t):
    """使用提取出的样本点计算指标，避免全量加载"""
    mask = ~np.isnan(p) & ~np.isnan(t)
    p, t = p[mask], t[mask]
    
    if len(p) < 100: return {k: np.nan for k in ["RMSE", "CC", "KGE", "BIAS(%)"]}

    rmse = np.sqrt(np.mean((p - t)**2))
    cc = np.corrcoef(p, t)[0, 1]
    
    std_p, std_t = np.std(p), np.std(t)
    mean_p, mean_t = np.mean(p), np.mean(t)
    
    alpha = std_p / (std_t + 1e-6)
    beta = mean_p / (mean_t + 1e-6)
    kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    bias = (np.sum(p - t) / (np.sum(t) + 1e-6)) * 100
    
    return {
        "RMSE": round(rmse, 3), "CC": round(cc, 3), 
        "KGE": round(kge, 3), "BIAS(%)": round(bias, 2)
    }

# ==========================================
# 2. 内存友好的随机采样提取函数
# ==========================================
def get_efficient_samples(ds_dict, ref_var, sample_n=200000):
    """
    核心优化：不再 flatten 全量数据，而是随机抽取 N 个坐标点提取数据
    """
    # 获取任意一个数据集的维度
    any_ds = next(iter(ds_dict.values()))
    shape = any_ds.shape # (time, lat, lon)
    
    # 随机生成坐标索引
    # 为了保证效率，我们直接生成一维索引再转换，或者分块抽样
    # 这里采用快速随机位置提取
    t_idx = np.random.randint(0, shape[0], sample_n)
    y_idx = np.random.randint(0, shape[1], sample_n)
    x_idx = np.random.randint(0, shape[2], sample_n)
    
    samples = {}
    for name, ds in ds_dict.items():
        # 使用 isel 批量提取特定点的值并立即转为 numpy
        # 这种方式只会将这 sample_n 个点读入内存
        samples[name] = ds.values[t_idx, y_idx, x_idx]
        
    return pd.DataFrame(samples).dropna()

# ==========================================
# 3. 分区处理
# ==========================================
def process_zone_low_memory(zone_id):
    config_file = f"config_z{zone_id}.json"
    if not os.path.exists(config_file): return []

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    zone_name = config['region']['climate_id_to_english'].get(str(zone_id), f"Zone_{zone_id}")
    print(f"\n[正在处理] {zone_id}: {zone_name} (低内存模式)")

    # 使用 chunks='auto' 开启 Dask 延迟加载，不占内存
    ds_ref = xr.open_dataset(config['data']['reference']['path'], chunks='auto')[config['data']['reference']['var']]
    
    # 建立待提取的数据字典
    data_to_sample = {"target": ds_ref}
    
    # 1. 原始产品
    prod_names = []
    for prod in config['data']['precip_products']:
        ds_raw = xr.open_dataset(prod['path'], chunks='auto')[prod['var']].reindex_like(ds_ref, method='nearest')
        data_to_sample[prod['name']] = ds_raw
        prod_names.append(prod['name'])

    # 2. 融合产品 (XPAC)
    fused_path = os.path.join(config['io']['out_dir'], config['io']['fused_nc_cnn'])
    if os.path.exists(fused_path):
        ds_xp = xr.open_dataset(fused_path, chunks='auto')
        var_name = list(ds_xp.data_vars)[0]
        data_to_sample["XPAC"] = ds_xp[var_name].reindex_like(ds_ref, method='nearest')

    # --- 执行采样 (关键步骤) ---
    print(f"  正在从 17 亿数据中提取样本点...")
    df_samples = get_efficient_samples(data_to_sample, "target", sample_n=500000)
    
    # 计算 Simple Average 样本
    df_samples['Simple Average'] = df_samples[prod_names].mean(axis=1)

    results = []
    # --- 评估各方法 ---
    methods = prod_names + ['Simple Average']
    if "XPAC" in df_samples.columns:
        methods.append("XPAC")

    for m_name in methods:
        m = calculate_metrics_from_samples(df_samples[m_name].values, df_samples['target'].values)
        m.update({"Zone_ID": zone_id, "Method": m_name})
        results.append(m)

    # --- 评估随机森林 (RF) ---
    print(f"  训练 RF 基准...")
    X = df_samples[prod_names]
    y = df_samples['target']
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    # 采样一部分用于训练，另一部分用于评估
    train_df = df_samples.sample(frac=0.5)
    test_df = df_samples.drop(train_df.index)
    rf.fit(train_df[prod_names], train_df['target'])
    
    rf_pred = rf.predict(test_df[prod_names])
    m_rf = calculate_metrics_from_samples(rf_pred, test_df['target'].values)
    m_rf.update({"Zone_ID": zone_id, "Method": "Random Forest"})
    results.append(m_rf)

    return results

if __name__ == "__main__":
    final_results = []
    for i in range(1, 8):
        res = process_zone_low_memory(i)
        final_results.extend(res)

    df = pd.DataFrame(final_results)
    df.to_csv("ablation_results_efficient.csv", index=False)
    print("\n[完成] 评估结果已保存至 ablation_results_efficient.csv")
    print(df.to_string(index=False))