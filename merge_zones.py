import os
import json
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
from precip_fusion_rf_twostage import load_config, build_climate_mask

def merge_all_zones(config_paths, final_out_path):
    """
    config_paths: list of str, 7个配置文件的路径
    final_out_path: str, 最终输出路径
    """
    
    # 1. 读取基础信息 (使用第一个配置文件的信息来加载 Mask)
    print("正在初始化基础网格与气候分区 Mask...")
    base_cfg = load_config(config_paths[0])
    
    # 临时加载一个 Reference Dataset 以获取 lat/lon
    ref_ds = xr.open_dataset(base_cfg["data"]["reference"]["path"]).isel(time=0)
    lat = ref_ds.lat.values
    lon = ref_ds.lon.values
    H, W = len(lat), len(lon)
    
    # 构建 Climate Mask (H, W)
    # 注意：这里会用到你配置里的中文映射字典
    name_map = base_cfg["region"].get("climate_name_map", None)
    climate_mask_da, _ = build_climate_mask(
        base_cfg["region"]["china_climate_shp"],
        base_cfg["region"]["climate_field"],
        ref_ds,
        name_map=name_map
    )
    climate_mask = climate_mask_da.values # numpy array
    
    # 2. 确定时间维度 (读取 Zone 1 的预测结果)
    # 假设所有分区的预测时间长度一致
    z1_out_dir = base_cfg["io"]["out_dir"]
    z1_nc_name = base_cfg["io"]["fused_nc_cnn"]
    z1_path = os.path.join(z1_out_dir, z1_nc_name)
    
    if not os.path.exists(z1_path):
        raise FileNotFoundError(f"Zone 1 result not found at {z1_path}. Did you run prediction?")
        
    with xr.open_dataset(z1_path) as ds:
        times = ds.time.values
        T = len(times)
        
    print(f"检测到时间步长: {T} 天")
    print(f"输出网格: {H} x {W}")
    
    # 3. 创建空画布 (Memory Map 以防内存溢出)
    # fused_array = np.full((T, H, W), np.nan, dtype=np.float32)
    # 为了更稳健，我们逐个分区读取并填入
    
    # 初始化一个全 NaN 的数组
    fused_array = np.full((T, H, W), np.nan, dtype=np.float32)

    # 4. 循环 7 个分区进行拼图
    for cfg_path in config_paths:
        cfg = load_config(cfg_path)
        z_id = int(cfg["training"]["target_zone_id"])
        
        # 构造该分区的预测文件路径
        pred_path = os.path.join(cfg["io"]["out_dir"], cfg["io"]["fused_nc_cnn"])
        print(f"正在合并 Zone {z_id} (来源: {pred_path})...")
        
        if not os.path.exists(pred_path):
            print(f"[Warning] Zone {z_id} 文件不存在，跳过！")
            continue
            
        # 读取预测数据
        with xr.open_dataset(pred_path) as ds_z:
            # 假设变量名是 pr_fused
            data_z = ds_z["pr_fused"].values
            
            # 生成该分区的 Mask (只保留 mask == z_id 的像素)
            # 使用 numpy 广播
            zone_mask_2d = (climate_mask == z_id)
            
            # 这里的逻辑是：
            # data_z 是全图大小 (T, H, W)，但只有 Zone z_id 区域有有效值(或者全图都有值)
            # 我们只把 Zone z_id 区域的值“抠”出来，贴到 fused_array 上
            
            # 方法 A: 循环时间步 (省内存)
            for t in range(T):
                # 取出当前时刻的画布
                canvas = fused_array[t]
                # 取出当前时刻的预测
                src = data_z[t]
                # 仅在 Mask 区域赋值
                canvas[zone_mask_2d] = src[zone_mask_2d]
                fused_array[t] = canvas

    # 5. 保存最终结果
    print(f"正在保存最终融合产品到: {final_out_path}")
    ds_out = xr.Dataset(
        data_vars={"pr_fused": (("time", "lat", "lon"), fused_array)},
        coords={"time": times, "lat": lat, "lon": lon}
    )
    
    ds_out["pr_fused"].attrs = {
        "description": "Multi-source Fused Precipitation (China Mosaic)",
        "method": "Attention-Guided CNN + Tweedie Loss (Zonal Training)",
        "units": "mm/day"
    }
    
    ds_out.to_netcdf(final_out_path, encoding={"pr_fused": {"zlib": True, "complevel": 4}})
    print("✅ 全部完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 传入所有 config 文件的路径，用空格分隔
    parser.add_argument("--configs", nargs='+', required=True, help="List of config json paths")
    parser.add_argument("--out", required=True, help="Final output filename")
    args = parser.parse_args()
    
    merge_all_zones(args.configs, args.out)