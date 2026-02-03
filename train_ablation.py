import os
import json
import copy
import torch
from xpac_model import XPAC_Ablation 

def run_experiments():
    # 遍历所有分区 (Zone 1 - Zone 7)
    for zone_id in range(1, 8):
        print(f"\n{'='*30}\n Processing Zone {zone_id} \n{'='*30}")
        
        config_path = f"config_z{zone_id}.json"
        if not os.path.exists(config_path):
            print(f"Skipping {config_path}: File not found.")
            continue

        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = json.load(f)

        # 定义我们要跑的两个对比实验
        # Exp 3 (XPAC) 你已经跑完了，所以这里只跑 Exp 1 和 Exp 2
        experiments = [
            {
                "id": 1, 
                "name": "Exp1_BaseCNN", 
                "use_cbam": False, 
                "data_source": "precip_products"  # 使用原始数据
            },
            {
                "id": 2, 
                "name": "Exp2_CNN_BC", 
                "use_cbam": False, 
                "data_source": "precip_products_Corrected" # 使用修正数据
            }
        ]

        for exp in experiments:
            exp_name = f"{exp['name']}_Z{zone_id}" # e.g., Exp1_BaseCNN_Z1
            output_nc_name = f"{exp_name}.nc"
            
            # 检查是否已经跑过，防止重复跑
            final_out_path = os.path.join(base_config['io']['out_dir'], output_nc_name)
            if os.path.exists(final_out_path):
                print(f"[Skip] {exp_name} already exists.")
                continue

            print(f">>> Running {exp_name} ...")

            # --- A. 动态配置修改 ---
            exp_config = copy.deepcopy(base_config)
            
            # 关键步骤：切换数据源
            # 如果 data_source 是 'precip_products_Corrected'，我们就把它覆盖到 'precip_products' 上
            # 这样后续的数据加载代码(DataLoader)不需要改动，因为它默认读 'precip_products'
            if exp['data_source'] == "precip_products_Corrected":
                if "precip_products_Corrected" in exp_config['data']:
                    print("    [Data] Switching to CORRECTED data...")
                    exp_config['data']['precip_products'] = exp_config['data']['precip_products_Corrected']
                else:
                    print(f"    [Error] 'precip_products_Corrected' not found in json for Zone {zone_id}!")
                    continue
            else:
                print("    [Data] Using RAW data...")

            # 设置输出文件名
            exp_config['io']['fused_nc_cnn'] = output_nc_name

            # --- B. 模型初始化 ---
            # 计算输入通道数
            in_channels = len(exp_config['data']['precip_products']) + \
                          len(exp_config['data']['static_covariates']) + \
                          len(exp_config['data']['dynamic_covariates'])
            
            model = XPAC_Ablation(in_channels=in_channels, use_cbam=exp['use_cbam']).cuda()
            
            # ==========================================================
            # TODO: 请在此处粘贴你的训练逻辑 (Data Loader, Loop, Save)
            # ==========================================================
            # 伪代码示例：
            # 1. 准备数据
            # train_loader = create_train_loader(exp_config)
            # test_loader = create_test_loader(exp_config)
            
            # 2. 训练
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            # for epoch in range(10): # 消融实验可以适当减少 epoch 比如跑 10-20 个
            #     train_epoch(model, train_loader, optimizer)
            
            # 3. 推理并保存 NC
            # save_inference_result(model, test_loader, final_out_path)
            # ==========================================================
            
            print(f"    [Done] Saved to {final_out_path}")

if __name__ == "__main__":
    run_experiments()