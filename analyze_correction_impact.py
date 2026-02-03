import os
import json
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与辅助函数
# ==========================================
# 定义参考数据路径 (用于所有分析的真值)
REF_PATH = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/CMFDV2.TIMEFIX.daily.CHINA.nc"

def get_paired_data(zone_id, sample_n=50000):
    """
    针对指定分区，提取成对的 (Reference, Raw, Corrected) 数据
    """
    config_file = f"config_z{zone_id}.json"
    if not os.path.exists(config_file): return None, None
    
    with open(config_file, 'r') as f:
        conf = json.load(f)
    
    # 1. 读取 Reference
    try:
        ds_ref = xr.open_dataset(REF_PATH, chunks='auto')
        # 自动找变量名
        var_ref = [v for v in ds_ref.data_vars if 'lat' not in v][0]
        # 简单随机采样 (为了画散点图，不用全量)
        # 这里为了速度，先读入内存再采样 (假设单分区 Reference 不大)
        # 如果内存不够，请参考之前的 chunk 采样法
        ref_all = ds_ref[var_ref].values.flatten()
        valid_idx = np.where(~np.isnan(ref_all))[0]
        
        if len(valid_idx) > sample_n:
            idx = np.random.choice(valid_idx, sample_n, replace=False)
        else:
            idx = valid_idx
            
        ref_sample = ref_all[idx]
    except Exception as e:
        print(f"Error reading Reference: {e}")
        return None, None

    # 2. 读取 Raw 和 Corrected (以 MSWX 为例，你可以循环做其他的)
    # 我们这里默认取列表里的第一个产品做展示 (通常是 MSWX)
    prod_raw = conf['data']['precip_products'][0]
    prod_corr = conf['data']['precip_products_Corrected'][0]
    
    print(f"Analyzing Zone {zone_id}: {prod_raw['name']} (Raw) vs {prod_corr['name']} (Corrected)")
    
    try:
        # Raw
        ds_raw = xr.open_dataset(prod_raw['path'], chunks='auto')[prod_raw['var']]
        raw_sample = ds_raw.values.flatten()[idx]
        
        # Corrected
        ds_corr = xr.open_dataset(prod_corr['path'], chunks='auto')[prod_corr['var']]
        corr_sample = ds_corr.values.flatten()[idx]
        
        # 构造 DataFrame
        df = pd.DataFrame({
            'Reference': ref_sample,
            'Raw': raw_sample,
            'Corrected': corr_sample
        })
        # 过滤负值 (修正过程可能产生微小负值) 和无效值
        df = df[df['Corrected'] >= 0].dropna()
        return df, prod_raw['name']
        
    except Exception as e:
        print(f"Error reading products: {e}")
        return None, None

# ==========================================
# 2. 绘图模块 (RSE 风格)
# ==========================================

def plot_cdf_comparison(df, model_name, zone_id, out_dir):
    """
    图 1: CDF 累积分布曲线
    用于证明: 校正后是否修正了降水频率分布 (Frequency Distribution)
    """
    plt.figure(figsize=(8, 6))
    
    # 筛选有效降水 (>0.1mm)
    df_wet = df[df['Reference'] > 0.1]
    
    sns.ecdfplot(data=df_wet, x='Reference', label='Reference', color='black', linewidth=2, linestyle='--')
    sns.ecdfplot(data=df_wet, x='Raw', label=f'Raw {model_name}', color='gray', linewidth=1.5)
    sns.ecdfplot(data=df_wet, x='Corrected', label=f'Corrected {model_name}', color='red', linewidth=1.5)
    
    plt.xscale('log') # 对数坐标看极端值
    plt.xlabel('Precipitation Intensity (mm/day)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'Zone {zone_id}: Improvement in Precipitation Distribution (CDF)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    save_path = os.path.join(out_dir, f"Fig_Correction_CDF_Z{zone_id}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

def plot_scatter_density(df, model_name, zone_id, out_dir):
    """
    图 2: 散点密度对比图 (Raw vs Corrected)
    用于证明: 系统性偏差 (Bias) 被消除，散点更集中于 1:1 线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    # 设置范围 (为了美观，截断极值)
    vmax = np.percentile(df['Reference'], 99.5)
    
    # 子图 1: Raw
    # 使用 hexbin 避免点太密看不清
    hb1 = axes[0].hexbin(df['Reference'], df['Raw'], gridsize=50, cmap='Blues', mincnt=1, bins='log', extent=[0, vmax, 0, vmax])
    axes[0].plot([0, vmax], [0, vmax], 'k--', lw=1)
    
    # 计算指标
    bias_raw = (df['Raw'].mean() - df['Reference'].mean()) / df['Reference'].mean() * 100
    cc_raw = np.corrcoef(df['Raw'], df['Reference'])[0, 1]
    axes[0].set_title(f"Before: Raw {model_name}\nBias: {bias_raw:.1f}%, CC: {cc_raw:.2f}")
    axes[0].set_xlabel("Reference (mm/day)")
    axes[0].set_ylabel("Raw Product (mm/day)")
    
    # 子图 2: Corrected
    hb2 = axes[1].hexbin(df['Reference'], df['Corrected'], gridsize=50, cmap='Reds', mincnt=1, bins='log', extent=[0, vmax, 0, vmax])
    axes[1].plot([0, vmax], [0, vmax], 'k--', lw=1)
    
    bias_corr = (df['Corrected'].mean() - df['Reference'].mean()) / df['Reference'].mean() * 100
    cc_corr = np.corrcoef(df['Corrected'], df['Reference'])[0, 1]
    axes[1].set_title(f"After: Phase 2 Correction\nBias: {bias_corr:.1f}%, CC: {cc_corr:.2f}")
    axes[1].set_xlabel("Reference (mm/day)")
    axes[1].set_ylabel("Corrected Product (mm/day)")
    
    cb = fig.colorbar(hb2, ax=axes, pad=0.02, fraction=0.05)
    cb.set_label('Point Count (Log Scale)')
    
    plt.suptitle(f"Zone {zone_id}: Scatter Density Comparison", fontsize=14)
    save_path = os.path.join(out_dir, f"Fig_Correction_Scatter_Z{zone_id}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

def plot_monthly_bias_boxplot(zone_id, out_dir):
    """
    图 3: 月度 Bias 箱线图
    用于证明: "Monthly Two-Stage" 方法不仅在全年有效，而且在每个月份都有效
    """
    # 这个函数需要重新快速读取全量数据的月均值 (采样不够准确)
    # 为了简化演示，这里我们假设上面的 sample 已经足够代表
    # 实际操作建议: 使用 pandas 的 datetime 索引进行 groupby
    pass 
    # (注: 由于上面是随机采样没带时间戳，这里略去具体代码。
    # 如果需要画这个，需要在 get_paired_data 里把时间也读出来)

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # 你可以选择只跑一个典型分区，比如 Zone 1
    for z in range(1, 8):
        df, name = get_paired_data(z, sample_n=50000)
        
        if df is not None:
            # 这里的 out_dir 从配置文件读取
            with open(f"config_z{z}.json") as f:
                out_dir = json.load(f)['io']['out_dir']
                
            plot_cdf_comparison(df, name, z, out_dir)
            plot_scatter_density(df, name, z, out_dir)