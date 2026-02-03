#!/bin/bash
set -e

zones=(1 2 3 4 5 6 7)
ks=(1 3 5 7 9 10)

for z in "${zones[@]}"; do
  for k in "${ks[@]}"; do
    cfg="configs_sens_tiles/config_z${z}_k${k}.json"
    python train_fusion_cnn.py --config "$cfg"
    python predict_fusion_cnn.py --config "$cfg"
  done
done
