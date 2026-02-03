#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Script for Attention-Guided Deep Learning Fusion (Zonal Version)
=========================================================================
功能：
1. 读取 NetCDF 数据（降水产品 + 协变量 + 气候分区掩膜）。
2. 根据配置文件中的 target_zone_id，只提取该分区的 Patch 进行训练。
3. 对数据进行标准化 (Z-Score)。
4. 使用 AttentionFusionNet + TweedieLoss 进行训练。
5. 保存模型权重 (.pth) 和统计参数 (stats.pkl)。
"""

import os
import json
import joblib
import argparse
import numpy as np
import xarray as xr
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 引入自定义模型和工具函数
# 确保 model_fusion.py 和 precip_fusion_rf_twostage.py 在同一目录下
from model_fusion import AttentionFusionNet, TweedieLoss
from precip_fusion_rf_twostage import load_config, load_all_inputs

# -----------------------------
# 1. Patch Dataset (支持分区采样)
# -----------------------------
class PrecipPatchDataset(Dataset):
    def __init__(self, inputs_dict, target_da, climate_mask, target_zone_id, patch_size=64, samples_per_epoch=4000):
        """
        Args:
            inputs_dict: 输入特征字典 {'precip1': (T,H,W), 'dem': (H,W), ...}
            target_da: 目标降水数组 (T,H,W)
            climate_mask: 气候分区掩膜 (H,W), 值为 1, 2, 3...
            target_zone_id: 当前训练的目标分区 ID (int)
            patch_size: 切片大小 (默认 64)
            samples_per_epoch: 每个 Epoch 采样的 Patch 数量
        """
        self.inputs = inputs_dict
        self.target = target_da
        self.patch_size = patch_size
        self.samples = samples_per_epoch
        
        # 获取维度信息
        self.T, self.H, self.W = target_da.shape
        
        # 排序 Key 以保证通道顺序固定
        self.keys = sorted(list(inputs_dict.keys()))
        
        # --- 核心：预计算有效采样点 ---
        # 我们只希望采样中心点位于目标分区内的 Patch
        # climate_mask 必须是 numpy array
        if climate_mask.shape != (self.H, self.W):
            raise ValueError(f"Climate mask shape {climate_mask.shape} does not match target shape {(self.H, self.W)}")

        print(f"Initializing dataset for Zone {target_zone_id}...")
        
        # 找到该分区的所有坐标点
        # valid_mask 是布尔矩阵
        valid_mask = (climate_mask == target_zone_id)
        
        # 为了防止 Patch 越界，我们在边界处留出 margin
        # 简单的做法：先找到所有有效点，然后在 __getitem__ 里做越界保护
        self.valid_ys, self.valid_xs = np.where(valid_mask)
        
        if len(self.valid_ys) == 0:
            raise ValueError(f"Zone {target_zone_id} has no valid pixels in the climate mask! Please check your mask or ID.")
            
        print(f"Zone {target_zone_id}: Found {len(self.valid_ys)} valid pixels for sampling.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # 1. 随机选择时间 t
        t = np.random.randint(0, self.T)
        
        # 2. 随机选择一个属于该分区的中心点 (Center Point)
        k = np.random.randint(0, len(self.valid_ys))
        ct_y, ct_x = self.valid_ys[k], self.valid_xs[k]
        
        # 3. 计算 Patch 左上角 (Top-Left)
        # 我们希望 (ct_y, ct_x) 包含在 Patch 内，所以随机偏移
        offset_y = np.random.randint(0, self.patch_size)
        offset_x = np.random.randint(0, self.patch_size)
        
        y = ct_y - offset_y
        x = ct_x - offset_x
        
        # 4. 越界修正 (Clip)
        # 保证 y, x 在 [0, H - patch_size] 范围内
        y = int(np.clip(y, 0, self.H - self.patch_size))
        x = int(np.clip(x, 0, self.W - self.patch_size))
        
        # 5. 构建 Input Tensor
        input_patches = []
        for key in self.keys:
            data = self.inputs[key]
            # 根据维度切片
            if data.ndim == 3: # (T, H, W)
                patch = data[t, y : y+self.patch_size, x : x+self.patch_size]
            elif data.ndim == 2: # (H, W)
                patch = data[y : y+self.patch_size, x : x+self.patch_size]
            else:
                raise ValueError(f"Unexpected data dim: {data.ndim}")
            
            input_patches.append(patch)
            
        # Stack -> (C, patch_size, patch_size)
        X = np.stack(input_patches, axis=0).astype(np.float32)
        
        # 6. 构建 Target
        Y = self.target[t, y : y+self.patch_size, x : x+self.patch_size].astype(np.float32)
        
        # 7. NaN 处理
        # CNN 输入不能有 NaN，填 0 (假设已标准化，0 代表均值)
        X = np.nan_to_num(X, nan=0.0)
        # Target 的 NaN 填 -1，在 Loss 中 Mask 掉
        Y = np.nan_to_num(Y, nan=-1.0)
        
        return torch.from_numpy(X), torch.from_numpy(Y)

# -----------------------------
# 2. 训练主流程
# -----------------------------
def train_cnn(cfg):
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 读取配置
    out_dir = cfg["io"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    # 获取目标分区 ID
    target_zone = cfg["training"].get("target_zone_id", None)
    if target_zone is None:
        raise ValueError("Config must specify 'target_zone_id' inside 'training' block!")
    target_zone = int(target_zone)
    print(f"Target Zone ID: {target_zone}")

    # --- Step 1: 加载所有数据 ---
    print("Loading data into memory (this may take time)...")
    # 调用 precip_fusion_rf_twostage.py 中的通用加载函数
    # 注意：该函数返回的 climate_id 是 (H, W) 的 DataArray
    ref_ds, ref_pr, prods, static_covs, dynamic_covs, climate_id, climate_mapping, times = load_all_inputs(cfg)
    
    # 提取 Climate Mask Numpy 数组
    climate_mask_arr = climate_id.values
    
    # --- Step 2: 数据标准化 (Standardization) ---
    print("Normalizing features...")
    data_dict = {}
    stats = {}
    
    def normalize_array(name, arr):
        # 计算均值方差 (忽略 NaN)
        mean = np.nanmean(arr)
        std = np.nanstd(arr) + 1e-5 # 防止除以0
        arr_norm = (arr - mean) / std
        stats[name] = {'mean': float(mean), 'std': float(std)}
        return arr_norm

    # 1. 降水产品
    for name, da in prods:
        print(f"  - Norm: {name}")
        data_dict[name] = normalize_array(name, da.values)
        
    # 2. 静态变量
    for name, da in static_covs:
        print(f"  - Norm: {name}")
        data_dict[name] = normalize_array(name, da.values)
        
    # 3. 动态变量
    for name, da in dynamic_covs:
        print(f"  - Norm: {name}")
        data_dict[name] = normalize_array(name, da.values)
        
    # 保存统计参数 (预测时需要用同样的参数反归一化/归一化)
    # 加上分区后缀，防止覆盖
    stats_path = os.path.join(out_dir, f"stats_zone_{target_zone}.pkl")
    joblib.dump(stats, stats_path)
    print(f"Saved stats to {stats_path}")
    
    # 保存通道名称顺序 (非常重要，预测时顺序必须一致)
    channel_names = sorted(list(data_dict.keys()))
    names_path = os.path.join(out_dir, f"channel_names_zone_{target_zone}.pkl")
    joblib.dump(channel_names, names_path)
    
    # 目标数据 (Target)
    target_arr = ref_pr.values
    
    # --- Step 3: 构建 DataLoader ---
    # 定义超参数
    patch_size = 64
    batch_size = cfg["training"].get("dl_batch_size", 32)
    samples_per_epoch = cfg["training"].get("samples_per_epoch", 4000)
    lr = cfg["training"].get("dl_lr", 1e-4)
    epochs = cfg["training"].get("dl_epochs", 50)

    dataset = PrecipPatchDataset(
        inputs_dict=data_dict,
        target_da=target_arr,
        climate_mask=climate_mask_arr,
        target_zone_id=target_zone,
        patch_size=patch_size,
        samples_per_epoch=samples_per_epoch
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # --- Step 4: 初始化模型 ---
    in_channels = len(channel_names)
    print(f"Model Input Channels: {in_channels}")
    print(f"Channels: {channel_names}")
    
    model = AttentionFusionNet(in_channels=in_channels, out_channels=1).to(device)
    
    # 使用 Tweedie Loss (p=1.5 适合降水这种半连续分布)
    criterion = TweedieLoss(p=1.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # --- Step 5: 训练循环 ---
    print("Start Training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred = model(X).squeeze(1) # (B, H, W)
            
            # Mask Loss: 忽略 Target 为 -1 (即 NaN/境外) 的像素
            # 只有当 Patch 里有有效值时才计算 Loss
            mask = (Y >= 0)
            if mask.sum() == 0:
                continue
                
            loss = criterion(pred[mask], Y[mask])
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected!")
                continue
                
            loss.backward()
            
            # 梯度裁剪，防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'loss': total_loss / (valid_batches + 1e-6)})
        
        avg_loss = total_loss / (valid_batches + 1e-6)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.6f}")
            
        # 每 10 个 Epoch 保存一次 Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(out_dir, f"fusion_zone_{target_zone}_ep{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            
    # 保存最终模型
    final_path = os.path.join(out_dir, f"fusion_zone_{target_zone}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Model saved to {final_path}")

# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    train_cnn(cfg)