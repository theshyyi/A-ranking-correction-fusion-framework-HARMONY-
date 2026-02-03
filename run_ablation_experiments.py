import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import warnings
from xpac_model import XPAC_Ablation 

# 忽略 xarray 序列化警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 辅助工具：Tweedie Loss (针对降水优化)
# ==========================================
class TweedieLoss(nn.Module):
    def __init__(self, p=1.5, reduction='mean'):
        super(TweedieLoss, self).__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred + 1e-8
        target = target + 1e-8
        term1 = (pred ** (2 - self.p)) / (2 - self.p)
        term2 = target * (pred ** (1 - self.p)) / (1 - self.p)
        loss = term1 - term2
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

# ==========================================
# 2. 数据集类：自动读取 NC 并对齐
# ==========================================
class ZoneDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        
        # 1. 读取参考数据
        ref_cfg = config['data']['reference']
        print(f"    [Dataset] Loading Reference: {ref_cfg['path']} ...")
        self.ds_ref = xr.open_dataset(ref_cfg['path'])[ref_cfg['var']]
        
        # 2. 读取输入产品 (Input Products) - 列表已被自动替换(Raw/Corrected)
        self.inputs = []
        for prod in config['data']['precip_products']:
            ds = xr.open_dataset(prod['path'])[prod['var']].reindex_like(self.ds_ref, method='nearest')
            self.inputs.append(ds)
            
        # 3. 读取静态协变量
        for cov in config['data']['static_covariates']:
            ds = xr.open_dataset(cov['path'])[cov['var']].reindex_like(self.ds_ref, method='nearest')
            ds_expanded = ds.expand_dims(time=self.ds_ref.time)
            self.inputs.append(ds_expanded)

        # 4. 读取动态协变量
        for cov in config['data']['dynamic_covariates']:
            ds = xr.open_dataset(cov['path'])[cov['var']].reindex_like(self.ds_ref, method='nearest')
            self.inputs.append(ds)
            
        # 5. 合并数据
        print("    [Dataset] Merging and aligning data...")
        ref_flat = self.ds_ref.values.flatten()
        input_flats = [x.values.flatten() for x in self.inputs]
        
        self.X = np.stack(input_flats, axis=1)
        self.y = ref_flat[:, np.newaxis]
        
        # 清洗 NaN
        valid_mask = ~np.isnan(self.X).any(axis=1) & ~np.isnan(self.y).any(axis=1)
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]
        
        # 简单类型转换 (建议生产环境增加标准化 Scaler)
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)
        
        print(f"    [Dataset] Ready. Samples: {self.X.shape[0]}, Features: {self.X.shape[1]}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# ==========================================
# 3. 主运行逻辑
# ==========================================
def run_experiments():
    # 遍历所有分区 (Zone 1 - Zone 7)
    for zone_id in range(1, 8):
        print(f"\n{'='*40}\n Processing Zone {zone_id} \n{'='*40}")
        
        config_path = f"config_z{zone_id}.json"
        if not os.path.exists(config_path):
            print(f"Skipping {config_path}: File not found.")
            continue

        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = json.load(f)

        # ==========================================
        # 定义完整的三个消融实验
        # ==========================================
        experiments = [
            {
                "id": 1, 
                "name": "Exp1_BaseCNN", 
                "use_cbam": False, 
                "data_source": "precip_products"  # 原始数据
            },
            {
                "id": 2, 
                "name": "Exp2_CNN_BC", 
                "use_cbam": False, 
                "data_source": "precip_products_Corrected" # 修正数据
            },
            {
                "id": 3, 
                "name": "Exp3_XPAC_CBAM", # 你的最终模型 (XPAC)
                "use_cbam": True,         # 开启注意力机制
                "data_source": "precip_products_Corrected" # 修正数据
            }
        ]

        for exp in experiments:
            exp_name = f"{exp['name']}_Z{zone_id}"
            output_nc_name = f"{exp_name}.nc"
            final_out_path = os.path.join(base_config['io']['out_dir'], output_nc_name)
            
            # 防止重复跑
            if os.path.exists(final_out_path):
                print(f"[Skip] {exp_name} already exists.")
                continue

            print(f">>> Running {exp_name} ...")

            # --- A. 动态配置修改 ---
            exp_config = copy.deepcopy(base_config)
            
            # 自动切换数据源
            if exp['data_source'] == "precip_products_Corrected":
                if "precip_products_Corrected" in exp_config['data']:
                    print("    [Data] Mode: Corrected Data")
                    exp_config['data']['precip_products'] = exp_config['data']['precip_products_Corrected']
                else:
                    print(f"    [Error] Corrected data config missing for Zone {zone_id}!")
                    continue
            else:
                print("    [Data] Mode: Raw Data")

            exp_config['io']['fused_nc_cnn'] = output_nc_name

            # --- B. 准备数据 ---
            try:
                dataset = ZoneDataset(exp_config)
            except Exception as e:
                print(f"    [Error] Failed to load data: {e}")
                continue

            batch_size = 2048 # 显存允许可调大
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

            # --- C. 模型初始化 ---
            in_channels = dataset.X.shape[1]
            model = XPAC_Ablation(in_channels=in_channels, use_cbam=exp['use_cbam']).cuda()
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = TweedieLoss(p=1.5)

            # --- D. 训练循环 ---
            # 为了消融实验的公平性，建议 Epoch 保持一致
            epochs = 5 
            print(f"    [Train] {exp_name} | CBAM={exp['use_cbam']} | Epochs={epochs}")
            
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                    # 适配 CNN 输入: [B, C] -> [B, C, 1, 1]
                    X_batch = X_batch.unsqueeze(-1).unsqueeze(-1)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch.squeeze())
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"        Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

            # --- E. 推理与保存 ---
            print("    [Inference] Generating full domain prediction...")
            model.eval()
            
            # 获取地理坐标参考
            ref_da = xr.open_dataset(exp_config['data']['reference']['path'])[exp_config['data']['reference']['var']]
            
            # 构造全量特征堆叠 (T, H, W, C)
            feature_stack = np.stack([ds.values for ds in dataset.inputs], axis=-1)
            T, H, W, C = feature_stack.shape
            feature_flat = feature_stack.reshape(-1, C)
            
            # 分批推理
            fused_flat = []
            infer_batch = 100000
            with torch.no_grad():
                for i in range(0, feature_flat.shape[0], infer_batch):
                    batch_data = feature_flat[i : i+infer_batch]
                    # 简单的 NaN 处理: 填 0 (或者用 mask 记录)
                    batch_tensor = torch.from_numpy(np.nan_to_num(batch_data)).float().cuda()
                    batch_tensor = batch_tensor.unsqueeze(-1).unsqueeze(-1)
                    
                    pred = model(batch_tensor)
                    fused_flat.append(pred.squeeze().cpu().numpy())
            
            fused_flat = np.concatenate(fused_flat)
            fused_volume = fused_flat.reshape(T, H, W)
            
            # 还原地理掩膜 (Mask)
            ref_mask = np.isnan(ref_da.values)
            fused_volume[ref_mask] = np.nan
            
            # 写入 NetCDF
            ds_out = xr.Dataset(
                {"precip": (("time", "lat", "lon"), fused_volume)},
                coords=ref_da.coords
            )
            ds_out.precip.attrs['units'] = 'mm/day'
            ds_out.attrs['experiment'] = exp['name']
            
            os.makedirs(base_config['io']['out_dir'], exist_ok=True)
            ds_out.to_netcdf(final_out_path)
            
            print(f"    [Done] Saved: {final_out_path}")
            
            # 清理显存
            del model, dataset, ds_out, fused_volume, feature_stack
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_experiments()