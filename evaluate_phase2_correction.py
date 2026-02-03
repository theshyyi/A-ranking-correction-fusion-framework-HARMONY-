import os
import json
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心评估指标计算 (KGE, RMSE, BIAS)
# ==========================================
def calculate_metrics(sim, obs):
    """
    计算一组模拟值(sim)与观测值(obs)的统计指标
    """
    # 剔除无效值
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    s = sim[mask]
    o = obs[mask]
    
    if len(s) < 100: return {k: np.nan for k in ["RMSE", "CC", "KGE", "BIAS"]}

    # RMSE
    rmse = np.sqrt(np.mean((s - o)**2))
    
    # CC
    cc = np.corrcoef(s, o)[0, 1]
    
    # KGE
    std_s, std_o = np.std(s), np.std(o)
    mean_s, mean_o = np.mean(s), np.mean(o)
    alpha = std_s / (std_o + 1e-6)
    beta = mean_s / (mean_o + 1e-6)
    kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    # Bias (%)
    bias = (np.sum(s - o) / (np.sum(o) + 1e-6)) * 100
    
    return {
        "RMSE": rmse,
        "CC": cc,
        "KGE": kge,
        "BIAS": bias
    }

# ==========================================
# 2. 采样加载器 (内存安全)
# ==========================================
def get_paired_samples(config, sample_n=200000):
    """
    从 Reference, Raw, Corrected 中提取成对的样本点
    """
    # 1. 打开 Reference
    ref_path = config['data']['reference']['path']
    try:
        ds_ref = xr.open_dataset(ref_path, chunks='auto')[config['data']['reference']['var']]
    except FileNotFoundError:
        print(f"Reference file not found: {ref_path}")
        return None

    # 生成随机坐标
    T, H, W = ds_ref.shape
    t_idx = np.random.randint(0, T, sample_n)
    y_idx = np.random.randint(0, H, sample_n)
    x_idx = np.random.randint(0, W, sample_n)
    
    # 构造索引器
    t_xr = xr.DataArray(t_idx, dims="sample")
    y_xr = xr.DataArray(y_idx, dims="sample")
    x_xr = xr.DataArray(x_idx, dims="sample")
    
    # 提取 Reference
    ref_vals = ds_ref.isel(time=t_xr, lat=y_xr, lon=x_xr).values
    
    # 准备数据容器
    data = {"Reference": ref_vals}
    
    # 提取 Raw Products
    raw_names = []
    for prod in config['data']['precip_products']:
        name = prod['name']
        try:
            ds = xr.open_dataset(prod['path'], chunks='auto')[prod['var']]
            # 假设已对齐，直接提取
            vals = ds.isel(time=t_xr, lat=y_xr, lon=x_xr).values
            data[f"Raw_{name}"] = vals
            raw_names.append(name)
        except Exception as e:
            print(f"Error loading Raw {name}: {e}")

    # 提取 Corrected Products
    if "precip_products_Corrected" in config['data']:
        for prod in config['data']['precip_products_Corrected']:
            name = prod['name'] # 名字通常和 Raw 一样 (e.g., MSWX)
            try:
                ds = xr.open_dataset(prod['path'], chunks='auto')[prod['var']]
                vals = ds.isel(time=t_xr, lat=y_xr, lon=x_xr).values
                data[f"Corrected_{name}"] = vals
            except Exception as e:
                print(f"Error loading Corrected {name}: {e}")
    else:
        print("Warning: No 'precip_products_Corrected' found in config.")
        
    return pd.DataFrame(data), raw_names

# ==========================================
# 3. 主程序
# ==========================================
def evaluate_correction():
    all_results = []
    
    for z in range(1, 8):
        print(f"\nEvaluating Zone {z}...")
        config_file = f"config_z{z}.json"
        if not os.path.exists(config_file): continue
        
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # 获取采样数据
        df, product_names = get_paired_samples(config)
        if df is None: continue
        
        # 计算指标
        for name in product_names:
            raw_col = f"Raw_{name}"
            corr_col = f"Corrected_{name}"
            
            if raw_col in df.columns and corr_col in df.columns:
                # 1. 计算 Raw 指标
                m_raw = calculate_metrics(df[raw_col].values, df["Reference"].values)
                m_raw.update({"Zone": z, "Product": name, "Type": "Raw"})
                all_results.append(m_raw)
                
                # 2. 计算 Corrected 指标
                m_corr = calculate_metrics(df[corr_col].values, df["Reference"].values)
                m_corr.update({"Zone": z, "Product": name, "Type": "Corrected"})
                all_results.append(m_corr)
                
                print(f"  {name}: KGE {m_raw['KGE']:.2f} -> {m_corr['KGE']:.2f} | Bias {m_raw['BIAS']:.1f}% -> {m_corr['BIAS']:.1f}%")

    # 保存结果
    res_df = pd.DataFrame(all_results)
    res_df.to_csv("phase2_correction_metrics.csv", index=False)
    print("\n结果已保存至 phase2_correction_metrics.csv")
    
    return res_df

# ==========================================
# 4. 绘图函数 (RSE 风格)
# ==========================================
def plot_correction_improvement(df):
    """绘制 Raw vs Corrected 的对比图"""
    if df.empty: return

    # 1. KGE 改进条形图
    plt.figure(figsize=(12, 6))
    
    # 数据转换：将 Raw 和 Corrected 并列
    sns.barplot(data=df, x='Zone', y='KGE', hue='Type', palette={'Raw': 'gray', 'Corrected': '#1f77b4'})
    
    plt.title("Effectiveness of Phase 2 Correction (KGE Improvement)", fontsize=14)
    plt.ylabel("Kling-Gupta Efficiency (KGE)")
    plt.xlabel("Climate Zone ID")
    plt.legend(title="Data Type")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig("Phase2_KGE_Comparison.png", dpi=300, bbox_inches='tight')
    print("图表已保存: Phase2_KGE_Comparison.png")

    # 2. Bias 缩减散点图
    plt.figure(figsize=(10, 6))
    
    # 过滤数据用于画图
    raw_df = df[df['Type'] == 'Raw']
    corr_df = df[df['Type'] == 'Corrected']
    
    # 绘制
    # 绝对 Bias 越接近 0 越好
    plt.scatter(raw_df['Zone'], raw_df['BIAS'].abs(), color='gray', label='Raw Bias', alpha=0.7, marker='o')
    plt.scatter(corr_df['Zone'], corr_df['BIAS'].abs(), color='red', label='Corrected Bias', alpha=0.7, marker='^')
    
    # 画箭头连接 Raw -> Corrected
    for i in range(len(raw_df)):
        r = raw_df.iloc[i]
        c = corr_df[(corr_df['Zone'] == r['Zone']) & (corr_df['Product'] == r['Product'])].iloc[0]
        plt.arrow(r['Zone'], abs(r['BIAS']), 0, abs(c['BIAS']) - abs(r['BIAS']), 
                  head_width=0.1, head_length=1, fc='k', ec='k', alpha=0.3, length_includes_head=True)

    plt.title("Reduction in Absolute Bias (%) after Correction", fontsize=14)
    plt.ylabel("Absolute Bias (%)")
    plt.xlabel("Climate Zone ID")
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("Phase2_Bias_Reduction.png", dpi=300, bbox_inches='tight')
    print("图表已保存: Phase2_Bias_Reduction.png")

if __name__ == "__main__":
    df_results = evaluate_correction()
    if not df_results.empty:
        plot_correction_improvement(df_results)