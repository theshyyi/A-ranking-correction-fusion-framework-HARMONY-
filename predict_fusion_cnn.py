import os
import joblib
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model_fusion import AttentionFusionNet
from precip_fusion_rf_twostage import load_config, load_all_inputs

def predict_sliding_window(model, input_tensor, patch_size=64, stride=48, device='cuda'):
    """
    input_tensor: (1, C, H, W)
    patch_size: 窗口大小
    stride: 步长 (小于 patch_size 以产生重叠)
    """
    _, C, H, W = input_tensor.shape
    output_map = torch.zeros((1, H, W), device=device)
    count_map = torch.zeros((1, H, W), device=device)
    
    # 简单的滑窗
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # 1. 切片
            patch = input_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
            # 2. 预测
            with torch.no_grad():
                pred = model(patch).squeeze(1) # (1, h, w)
            
            # 3. 累加到输出图
            output_map[:, y:y+patch_size, x:x+patch_size] += pred
            count_map[:, y:y+patch_size, x:x+patch_size] += 1.0
            
    # 处理边缘 (简单起见，这里忽略最右下角的残余部分，或者你可以用 pad)
    # 计算平均值
    output_map = output_map / (count_map + 1e-6)
    return output_map

def predict_cnn(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["io"]["out_dir"]
    
        # 获取目标分区 ID
    target_zone = cfg["training"].get("target_zone_id", None)
    if target_zone is None:
        raise ValueError("Config must specify 'target_zone_id' inside 'training' block!")
    target_zone = int(target_zone)
    print(f"Target Zone ID: {target_zone}")
    
    # 1. 加载元数据
    stats = joblib.load(os.path.join(out_dir, f"stats_zone_{target_zone}.pkl"))
    channel_names = joblib.load(os.path.join(out_dir, f"channel_names_zone_{target_zone}.pkl"))
    
    # 2. 加载数据
    print("Loading input data...")
    ref_ds, _, prods, static_covs, dynamic_covs, _, _, times = load_all_inputs(cfg)
    
    # 3. 加载模型
    model = AttentionFusionNet(in_channels=len(channel_names), out_channels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, f"fusion_zone_{target_zone}_final.pth")))
    model.eval()
    
    # 4. 准备输出
    lat = ref_ds["lat"].values
    lon = ref_ds["lon"].values
    fused_results = []
    
    # 5. 逐日预测 (Time Loop)
    print("Start predicting...")
    
    # 这里的 Batch 是时间维度
    for t_idx, t_val in enumerate(tqdm(times)):
        # --- 构建当前时刻的 Input Tensor ---
        input_list = []
        for name in channel_names:
            # 查找该变量在哪个列表里
            arr = None
            
            # 检查 Products
            for pname, da in prods:
                if pname == name:
                    arr = da.isel(time=t_idx).values
                    break
            # 检查 Static
            if arr is None:
                for sname, da in static_covs:
                    if sname == name:
                        arr = da.values # 2D
                        break
            # 检查 Dynamic
            if arr is None:
                for dname, da in dynamic_covs:
                    if dname == name:
                        arr = da.isel(time=t_idx).values
                        break
            
            if arr is None:
                raise ValueError(f"Feature {name} not found!")
                
            # 标准化
            mean = stats[name]['mean']
            std = stats[name]['std']
            arr = (arr - mean) / std
            arr = np.nan_to_num(arr, nan=0.0)
            input_list.append(arr)
            
        # Stack -> (1, C, H, W)
        img = np.stack(input_list, axis=0)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
        
        # 滑窗预测
        patch_size = int(cfg.get("prediction", {}).get("patch_size", 64))
        stride = int(cfg.get("prediction", {}).get("stride", 32))
        out_tensor = predict_sliding_window(model, img_tensor, patch_size=patch_size, stride=stride, device=device)

        
        fused_results.append(out_tensor.cpu().numpy().squeeze())

    # 6. 保存为 NetCDF
    fused_arr = np.stack(fused_results, axis=0) # (T, H, W)
    
    ds_out = xr.Dataset(
        data_vars={"pr_fused": (("time", "lat", "lon"), fused_arr.astype(np.float32))},
        coords={"time": times, "lat": lat, "lon": lon}
    )
    
    out_nc = os.path.join(out_dir, cfg["io"]["fused_nc_cnn"])
    ds_out.to_netcdf(out_nc, encoding={"pr_fused": {"zlib": True, "complevel": 4}})
    print(f"Saved: {out_nc}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    predict_cnn(cfg)