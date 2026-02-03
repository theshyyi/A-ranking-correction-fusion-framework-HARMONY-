import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# 配置参考数据路径
REF_PATH = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/CMFDV2.TIMEFIX.daily.CHINA.nc"

def get_valid_precip(nc_path, threshold=0.1):
    """读取 NC 并提取有效降水值 (用于PDF)"""
    if not os.path.exists(nc_path):
        return np.array([])
    try:
        with xr.open_dataset(nc_path) as ds:
            # 自动寻找变量名：排除 lat, lon, time
            var = [v for v in ds.data_vars if v not in ['lat', 'lon', 'time', 'spatial_ref']][0]
            val = ds[var].values.flatten()
            return val[val > threshold]
    except Exception as e:
        # print(f"Error reading {nc_path}: {e}")
        return np.array([])

def plot_zone_pdf(zone_id):
    config_file = f"config_z{zone_id}.json"
    if not os.path.exists(config_file): return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    
    out_dir = conf['io']['out_dir']
    
    # 定义三个文件的路径
    # Exp 1 & 2 是我们刚跑出来的
    # Exp 3 是你原本有的 (从 json 读取文件名)
    files = {
        "Exp1": os.path.join(out_dir, f"Exp1_BaseCNN_Z{zone_id}.nc"),
        "Exp2": os.path.join(out_dir, f"Exp2_CNN_BC_Z{zone_id}.nc"),
        "Exp3": os.path.join(out_dir, conf['io']['fused_nc_cnn']) 
    }
    
    print(f"Plotting PDF for Zone {zone_id}...")
    
    # 读取数据
    data_ref = get_valid_precip(REF_PATH)
    d1 = get_valid_precip(files["Exp1"])
    d2 = get_valid_precip(files["Exp2"])
    d3 = get_valid_precip(files["Exp3"])
    
    if len(data_ref) == 0:
        print("Reference data not found.")
        return

    # 开始绘图
    plt.figure(figsize=(10, 7))
    
    # 1. Reference (黑色虚线)
    sns.kdeplot(data_ref, color='black', linestyle='--', label='Reference (CMFD)', linewidth=2.5)
    
    # 2. Exp 1: Raw + Base CNN (蓝色)
    if len(d1) > 0:
        sns.kdeplot(d1, color='#1f77b4', label='Exp1: Base CNN (Raw)', linewidth=1.5, alpha=0.8)
    
    # 3. Exp 2: Corrected + Base CNN (橙色)
    if len(d2) > 0:
        sns.kdeplot(d2, color='#ff7f0e', label='Exp2: CNN + BiasCorrection', linewidth=1.5, alpha=0.8)
        
    # 4. Exp 3: XPAC (红色填充) - 你的最终结果
    if len(d3) > 0:
        sns.kdeplot(d3, color='#d62728', label='Exp3: XPAC (Final)', linewidth=2.5, fill=True, alpha=0.1)
        
    # 样式调整
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0.1, 200) # 0.1mm - 200mm
    plt.ylim(1e-4, 1)
    
    plt.title(f'Zone {zone_id}: Ablation Study - Extreme Precipitation PDF', fontsize=14)
    plt.xlabel('Precipitation Intensity (mm/day)', fontsize=12)
    plt.ylabel('Probability Density (Log Scale)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    save_path = f"RSE_Ablation_PDF_Zone{zone_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 循环画所有 7 个区
    for z in range(1, 8):
        plot_zone_pdf(z)