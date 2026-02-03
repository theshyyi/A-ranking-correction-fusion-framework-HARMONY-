import os
import joblib
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import xarray as xr
from model_fusion import AttentionFusionNet
from precip_fusion_rf_twostage import load_config, load_all_inputs
from train_fusion_cnn import PrecipPatchDataset
import torch.nn as nn
# 配置 matplotlib 支持中文 (为了画论文图)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Arial' 用于英文论文
plt.rcParams['axes.unicode_minus'] = False




import torch.nn as nn  # 确保导入 nn

# ---------------------------------------------------------
# 新增：SHAP 输出包装器
# 作用：将模型的二维输出 (H, W) 转换为标量 (Scalar)，
# 这样 SHAP 才能计算“特征对该区域总降水量的贡献”。
# ---------------------------------------------------------
class SHAPOutputWrapper(nn.Module):
    def __init__(self, model):
        super(SHAPOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 1. 获取原始模型输出 (Batch, 1, H, W)
        out = self.model(x)
        
        # 2. 核心修改：对空间维度求和，得到 (Batch, 1)
        # 物理含义：解释的是“该 Patch 内的总降水量”
        return out.view(out.shape[0], -1).sum(dim=1, keepdim=True)








def explain_model_shap(cfg):
    # ---------------------------------------------------------
    # 1. 准备环境与模型
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["io"]["out_dir"]
    target_zone_id = cfg["training"].get("target_zone_id")
    # 加载元数据
    stats = joblib.load(os.path.join(out_dir, f"stats_zone_{target_zone_id}.pkl"))
    channel_names = joblib.load(os.path.join(out_dir, f"channel_names_zone_{target_zone_id}.pkl"))
    
    # 加载模型
    print("Loading Model...")
    model = AttentionFusionNet(in_channels=len(channel_names), out_channels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, f"fusion_zone_{target_zone_id}_final.pth")))
    model.eval()
    
    
    # 1. 包装模型
    print("Wrapping model for SHAP output compatibility...")
    model_shap = SHAPOutputWrapper(model)
    model_shap.eval()
    
    # ---------------------------------------------------------
    # 2. 准备数据 (Background & Test)
    # ---------------------------------------------------------
    # SHAP DeepExplainer 需要一个 "Background" 数据集来作为基准
    # 我们随机抽取 100 个 patch 作为背景
    
    print("Preparing Data for SHAP...")
    # 这里为了演示，我们重新加载数据（或者你可以保存一些npy文件专门用于解释）
    # 假设我们只取一小部分用于演示，避免内存爆炸
    ref_ds, ref_pr, prods, static_covs, dynamic_covs, climate_id, climate_mapping, times = load_all_inputs(cfg)
    
    # 简单的归一化辅助函数
    def get_norm_data(name, da, time_idx=0):
        if name in [p[0] for p in prods]: # Products
            val = next(d for n, d in prods if n==name).isel(time=time_idx).values
        elif name in [s[0] for s in static_covs]: # Static
            val = next(d for n, d in static_covs if n==name).values
        elif name in [d[0] for d in dynamic_covs]: # Dynamic
            val = next(d for n, d in dynamic_covs if n==name).isel(time=time_idx).values
        else:
            raise ValueError(f"Unknown feature {name}")
            
        mean = stats[name]['mean']
        std = stats[name]['std']
        return np.nan_to_num((val - mean) / std, nan=0.0)

    # 选取一个特定的时间点（例如某次暴雨）作为 Test
    # 假设第 100 天有暴雨
    target_time_idx = 100 
    print(f"Analyzing event at time index: {target_time_idx}")

    # ---------------------------------------------------------
    # 智能 Patch 选择 (严格限制在 Target Zone 内 + 边界保护)
    # ---------------------------------------------------------
    patch_size = 64
    
    # 1. 获取目标分区 ID (从 config_z1.json 中读取)
    # config_z1.json -> "training": { "target_zone_id": 1 }
    if target_zone_id is None:
        raise ValueError("Config defines no 'target_zone_id' in ['training'] section!")
    
    print(f"Searching for storm center within Climate Zone ID: {target_zone_id}...")

    # 2. 获取参考降水数据 (Truth) 和 气候分区掩膜
    # load_all_inputs 返回了 ref_aligned (DataArray) 和 climate_id (DataArray)
    # 确保它们已经对齐
    ref_data_2d = ref_pr.isel(time=target_time_idx).values  # (Lat, Lon)
    climate_mask_2d = climate_id.values                     # (Lat, Lon), 值为 1.0, 2.0 等

    # 3. 构建搜索地图 (Search Map)
    # 逻辑：只保留 target_zone_id 内的降水，其他区域(包括 NaN 和其他分区)全部设为 -1
    # 这样 np.argmax 就绝对不会选到分区外面的点
    
    # 初始化全为 -1
    search_map = np.full_like(ref_data_2d, -1.0, dtype=np.float32)
    
    # 找到属于当前分区的索引
    # 注意：climate_mask_2d 可能是 float (1.0)，比较时要小心
    is_in_zone = np.isclose(climate_mask_2d, float(target_zone_id))
    
    # 将分区内的真实降水填入搜索地图 (同时处理 NaN，防止 NaN 干扰 argmax)
    # np.nan_to_num 将分区内的 NaN 降水也转为 0 (或其他负数)，避免报错
    valid_precip = np.nan_to_num(ref_data_2d, nan=-1.0)
    
    # 赋值：只有 mask 为 True 的地方才有值
    search_map[is_in_zone] = valid_precip[is_in_zone]

    # 4. 寻找最大值点 (argmax)
    flat_idx = np.argmax(search_map)
    max_y, max_x = np.unravel_index(flat_idx, search_map.shape)
    
    max_val = search_map[max_y, max_x]
    
    # 检查是否真的找到了有效点
    if max_val < 0:
        raise RuntimeError(f"Could not find any valid precipitation grid in Zone {target_zone_id} for time index {target_time_idx}!")
        
    print(f"Detected Storm Center in Zone {target_zone_id} at (y={max_y}, x={max_x}) with value: {max_val:.2f} mm")

    # 5. 计算 Patch 的左上角坐标 (cy, cx) - 带边界保护 (Clamping)
    # 这一步保持不变，确保切片不出界
    H, W = ref_data_2d.shape
    half_size = patch_size // 2
    
    # 纵向 (Y) 边界处理
    cy = max_y - half_size
    if cy < 0: 
        cy = 0
    elif cy + patch_size > H: 
        cy = H - patch_size
        
    # 横向 (X) 边界处理
    cx = max_x - half_size
    if cx < 0: 
        cx = 0
    elif cx + patch_size > W: 
        cx = W - patch_size

    print(f"Final Patch Top-Left: (y={cy}, x={cx}) -> Size: {patch_size}x{patch_size}")
    
    input_list = []
    for name in channel_names:
        full_map = get_norm_data(name, None, time_idx=target_time_idx)
        patch = full_map[cy:cy+patch_size, cx:cx+patch_size]
        input_list.append(patch)
        
    test_tensor = torch.from_numpy(np.stack(input_list)).unsqueeze(0).float().to(device) # (1, C, H, W)

    # 构建 Background Tensor (需要多张图，例如随机取 50 张不同时间的图)
    # SHAP 将对比 Test 图和 Background 图的差异来计算贡献
    bg_list = []
    for i in range(10): # 取 10 个样本作为背景 (实际论文建议 100+)
        rand_t = np.random.randint(0, len(times))
        tmp_list = []
        for name in channel_names:
            full_map = get_norm_data(name, None, time_idx=rand_t)
            # 随机切
            ry = np.random.randint(0, full_map.shape[0]-patch_size)
            rx = np.random.randint(0, full_map.shape[1]-patch_size)
            tmp_list.append(full_map[ry:ry+patch_size, rx:rx+patch_size])
        bg_list.append(np.stack(tmp_list))
    
    background_tensor = torch.from_numpy(np.stack(bg_list)).float().to(device) # (10, C, H, W)

    # ---------------------------------------------------------
    # 3. 计算 SHAP 值
    # ---------------------------------------------------------
    print("Calculating SHAP values (this may take a while)...")
    
    # 使用 DeepExplainer (适用于深度学习)
    # explainer = shap.DeepExplainer(model, background_tensor)
    
    # 2. 使用包装后的模型初始化 Explainer
    # 注意：这里传入的是 model_shap，而不是原始的 model
    explainer = shap.DeepExplainer(model_shap, background_tensor)
    # 计算 test_tensor 的 SHAP 值
    # 输出形状: list of tensors (因为模型可能多输出), 这里我们是一个输出
    # 修改后：添加 check_additivity=False
    shap_values = explainer.shap_values(test_tensor, check_additivity=False)
    
    # shap_values 是一个 list，对应模型的每个输出节点。
    # 因为我们输出只有一个通道 (降水)，所以取 shap_values[0] (或者它本身就是数组，取决于 shap 版本)
    if isinstance(shap_values, list):
        shap_vals = shap_values[0] # (1, C, H, W)
    else:
        shap_vals = shap_values

    # ---------------------------------------------------------
    # 4. 可视化与分析 (论文作图核心)
    # ---------------------------------------------------------
    
    # 1. 如果 shap_vals 是 list (GradientExplainer有时候返回list), 取出来
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
        
    # 2. 确保它转回 numpy (如果是tensor的话)
    if hasattr(shap_vals, 'cpu'):
        shap_vals = shap_vals.cpu().detach().numpy()
    elif not isinstance(shap_vals, np.ndarray):
        shap_vals = np.array(shap_vals)

    # 3. 计算全局平均 SHAP
    # 注意：shap_vals 形状通常是 (1, Channels, H, W) 或者 (batch, Channels, H, W)
    # 我们需要对 (Batch, H, W) 求平均，只保留 Channels 维度
    
    # ================= 修改开始 =================
    # 这里的 axis 需要根据 shap_vals 的实际维度来定
    # 假设 shap_vals 是 (1, 16, 64, 64) -> 我们希望得到 (16,)
    
    # 稳健的写法：直接展平为 (Channels, -1) 然后求均值
    shap_vals_flat = shap_vals.reshape(shap_vals.shape[1], -1)  # (Channels, Pixels)
    mean_shap = np.abs(shap_vals_flat).mean(axis=1)             # (Channels,)
    
    # 再次确保它是纯净的 numpy 数组，并且维度正确
    mean_shap = np.array(mean_shap).flatten() 
    # ================= 修改结束 =================

    print(f"SHAP shape: {shap_vals.shape}, Mean SHAP shape: {mean_shap.shape}") # 调试打印

    # --- 图表 1: 全局特征重要性 ---
    plt.figure(figsize=(10, 6))
    plt.barh(channel_names, mean_shap, color='skyblue')
    plt.xlabel('Average Absolute SHAP Value (Impact on Precip)')
    plt.title('Feature Importance for Selected Rain Event')
    plt.gca().invert_yaxis() # 最高的在上面
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_feature_importance.png"), dpi=300)
    print("Saved feature importance plot.")

    # --- 图表 2: 空间归因图 (Spatial Attribution Map) ---
    # 我们选几个关键通道展示：降水产品A、地形DEM、风速
    # 展示：原始输入 vs SHAP Heatmap
    
    input_img = test_tensor.cpu().numpy().squeeze() # (C, H, W)
    shap_img = shap_vals.squeeze() # (C, H, W)
    
    # 挑选前 4 个最重要的特征进行展示
    top_indices = np.argsort(mean_shap)[::-1][:4]
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    # 列定义: 1.原始输入特征, 2.SHAP贡献图, 3.最终融合降水
    
    # 获取模型预测结果用于对比
    with torch.no_grad():
        pred_precip = model(test_tensor).cpu().numpy().squeeze()
        
    for i, idx in enumerate(top_indices):
        name = channel_names[idx]
        
        # Col 1: Feature Value
        im1 = axes[i, 0].imshow(input_img[idx], cmap='viridis')
        axes[i, 0].set_title(f"Input: {name}")
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Col 2: SHAP Value (Red=Positive, Blue=Negative)
        # 使用 max_val 保证所有图色标一致，方便对比
        # Col 2: SHAP Value (Red=Positive, Blue=Negative)
        # 使用 99% 分位数作为最大值，剔除极值点的影响，让红蓝对比更明显
        max_val = np.percentile(np.abs(shap_img), 99) 
        
        # 防止 max_val 也就是全是0的情况导致报错
        if max_val < 1e-4: 
            max_val = 1.0

        # 这里保持不变
        im2 = axes[i, 1].imshow(shap_img[idx], cmap='seismic', vmin=-max_val, vmax=max_val)
        axes[i, 1].set_title(f"SHAP: {name} Contribution")
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Col 3: Prediction (只在第一行画，或者画点别的)
        if i == 0:
            im3 = axes[i, 2].imshow(pred_precip, cmap='Blues')
            axes[i, 2].set_title("Model Prediction (Fused)")
            plt.colorbar(im3, ax=axes[i, 2])
        else:
            axes[i, 2].axis('off') # 后面几行这一列留白
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_spatial_analysis.png"), dpi=300)
    print("Saved spatial analysis plot.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    explain_model_shap(cfg)