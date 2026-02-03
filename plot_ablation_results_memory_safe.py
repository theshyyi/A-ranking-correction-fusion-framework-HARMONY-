import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import gc # 垃圾回收模块

# ================= 配置区域 =================
# 参考数据路径
REF_PATH = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/CMFDV2.TIMEFIX.daily.CHINA.nc"
# 最终画图时保留的最大采样点数 (50万点对于KDE曲线足够平滑且精确)
MAX_SAMPLES_FOR_PLOT = 500000 
# ===========================================

def get_valid_precip_safe(nc_path, label, threshold=0.1):
    """
    内存安全的数据读取函数：
    采用分块读取 + 随机蓄水池采样策略
    """
    if not os.path.exists(nc_path):
        print(f"Warning: File not found {nc_path}")
        return np.array([])
    
    print(f"[{label}] Reading in chunks from: {os.path.basename(nc_path)}...")
    
    collected_samples = []
    total_collected = 0
    
    try:
        # 1. 使用 chunks='auto' 延迟打开，不立即加载数据
        ds = xr.open_dataset(nc_path, chunks='auto')
        
        # 自动获取变量名
        var_name = [v for v in ds.data_vars if v not in ['lat', 'lon', 'time', 'spatial_ref']][0]
        data_var = ds[var_name]
        
        # 2. 按时间分块处理 (每次处理 100 个时间步)
        # 这样内存峰值只取决于单次 Chunk 的大小
        time_size = data_var.shape[0]
        chunk_size = 100 
        
        for i in range(0, time_size, chunk_size):
            # 切片读取，此时才真正触发 IO 加载进内存
            subset = data_var.isel(time=slice(i, i + chunk_size)).values.flatten()
            
            # 提取有效降水 (> 0.1)
            # 过滤 NaN 和 0
            valid_mask = (subset > threshold) & (~np.isnan(subset))
            valid_vals = subset[valid_mask]
            
            # 3. 块内采样
            # 如果这一个块里的有效点太多，我们只取一部分，防止 collected_samples 爆炸
            # 假设我们最终想要 MAX_SAMPLES，分摊到每个块 (粗略估计)
            # 这里简单处理：每个块最多留 50000 个点
            if len(valid_vals) > 50000:
                valid_vals = np.random.choice(valid_vals, 50000, replace=False)
            
            if len(valid_vals) > 0:
                collected_samples.append(valid_vals)
                total_collected += len(valid_vals)
            
            # 打印进度
            # if i % 1000 == 0:
            #     print(f"    Processed {i}/{time_size} steps...")

        ds.close()
        
    except Exception as e:
        print(f"Error reading {nc_path}: {e}")
        return np.array([])
    
    # 4. 合并所有块
    if total_collected == 0:
        return np.array([])
    
    full_array = np.concatenate(collected_samples)
    
    # 5. 最终全局采样
    # 如果总数超过了画图需求，再次随机采样
    if len(full_array) > MAX_SAMPLES_FOR_PLOT:
        print(f"    Downsampling from {len(full_array)} to {MAX_SAMPLES_FOR_PLOT} for plotting...")
        final_array = np.random.choice(full_array, MAX_SAMPLES_FOR_PLOT, replace=False)
    else:
        final_array = full_array
        
    print(f"    -> Loaded {len(final_array)} points. Memory cleared.")
    
    # 手动触发垃圾回收
    del collected_samples, full_array
    gc.collect()
    
    return final_array

def plot_zone_pdf_safe(zone_id):
    config_file = f"config_z{zone_id}.json"
    if not os.path.exists(config_file): return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    
    out_dir = conf['io']['out_dir']
    
    # 定义文件路径
    files = {
        "Exp1": os.path.join(out_dir, f"Exp1_BaseCNN_Z{zone_id}.nc"),
        "Exp2": os.path.join(out_dir, f"Exp2_CNN_BC_Z{zone_id}.nc"),
        "Exp3": os.path.join(out_dir, conf['io']['fused_nc_cnn']) 
    }
    
    print(f"\n=== Plotting PDF for Zone {zone_id} (Memory Safe Mode) ===")
    
    # 1. 逐个读取数据 (读完一个扔一个，保持内存清爽)
    data_ref = get_valid_precip_safe(REF_PATH, "Reference")
    d1 = get_valid_precip_safe(files["Exp1"], "Exp1")
    d2 = get_valid_precip_safe(files["Exp2"], "Exp2")
    d3 = get_valid_precip_safe(files["Exp3"], "Exp3")
    
    if len(data_ref) == 0:
        print("Skipping plot due to missing reference data.")
        return

    # 2. 开始绘图
    plt.figure(figsize=(10, 7))
    
    # Reference
    sns.kdeplot(data_ref, color='black', linestyle='--', label='Reference (CMFD)', linewidth=2.5)
    
    # Exp 1
    if len(d1) > 0:
        sns.kdeplot(d1, color='#1f77b4', label='Exp1: Base CNN (Raw)', linewidth=1.5, alpha=0.8)
    
    # Exp 2
    if len(d2) > 0:
        sns.kdeplot(d2, color='#ff7f0e', label='Exp2: CNN + BiasCorrection', linewidth=1.5, alpha=0.8)
        
    # Exp 3
    if len(d3) > 0:
        sns.kdeplot(d3, color='#d62728', label='Exp3: XPAC (Final)', linewidth=2.5, fill=True, alpha=0.1)
        
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0.1, 200) 
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
    
    # 画完一张图，再次清理内存
    del data_ref, d1, d2, d3
    gc.collect()

if __name__ == "__main__":
    # 循环画所有 7 个区
    for z in range(1, 8):
        plot_zone_pdf_safe(z)