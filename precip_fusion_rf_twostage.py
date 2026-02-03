#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RF Two-stage precipitation fusion over China using gridded reference (CMFDV2)
===========================================================================
- Inputs: 3 corrected precip products (pr, mm/day) + static + dynamic covariates
- Target: reference gridded precip (CMFDV2.pr, mm/day)
- Region: China mask & 7 climate zones from shapefile (field: climate)

Run:
  Train:
    python precip_fusion_rf_twostage.py train --config /path/config_fusion_rf.json
  Predict:
    python precip_fusion_rf_twostage.py predict --config /path/config_fusion_rf.json
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import joblib
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# Utils
# -----------------------------
def load_config(path: str) -> dict:
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def ensure_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for k in list(ds.coords):
        lk = k.lower()
        if lk in ["latitude", "y"] and "lat" not in ds.coords:
            rename[k] = "lat"
        if lk in ["longitude", "x"] and "lon" not in ds.coords:
            rename[k] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds
def normalize_time(ds: xr.Dataset) -> xr.Dataset:
    """
    将各种 cftime / object time 统一转成 pandas datetime64[ns]，
    并归一到日尺度（00:00:00），保证不同数据源时间可求交集。
    """
    if "time" not in ds.coords:
        return ds

    t = ds["time"].values

    # 统一用字符串解析，兼容 cftime（DatetimeProlepticGregorian/DatetimeGregorian 等）
    # 只取 YYYY-MM-DD，忽略 10:30:00 这种时刻差异
    t_str = [str(x)[:10] for x in t]
    t2 = pd.to_datetime(t_str, errors="coerce")

    if pd.isna(t2).any():
        bad_idx = np.where(pd.isna(t2))[0][:10]
        bad_vals = [t_str[i] for i in bad_idx]
        raise ValueError(f"Failed to parse some time values to datetime. Examples: {bad_vals}")

    # 归一化到日尺度
    t2 = pd.DatetimeIndex(t2).normalize()

    return ds.assign_coords(time=t2)


def open_ds(path, chunks=None) -> xr.Dataset:
    ds = xr.open_dataset(path, chunks=chunks)
    ds = ensure_latlon(ds)
    ds = normalize_time(ds)   # <<<<<< 关键：统一 time
    return ds


def pick_var(ds: xr.Dataset, var: str | None, fallback_name: str | None = None) -> str:
    if var and var in ds.data_vars:
        return var
    if fallback_name and fallback_name in ds.data_vars:
        return fallback_name
    return list(ds.data_vars.keys())[0]


def align_grid(src: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    if np.array_equal(src["lat"].values, target["lat"].values) and np.array_equal(src["lon"].values, target["lon"].values):
        return src
    return src.interp(lat=target["lat"], lon=target["lon"], method="linear")


def time_intersection(datasets):
    times = None
    for ds in datasets:
        if "time" in ds.coords:
            t = pd.DatetimeIndex(ds["time"].values).normalize()
            times = t if times is None else times.intersection(t)

    if times is None or len(times) == 0:
        raise ValueError("time intersection is empty. Please check time axes.")

    times = pd.DatetimeIndex(times).sort_values()

    out = []
    for ds in datasets:
        if "time" in ds.coords:
            out.append(ds.sel(time=times))
        else:
            out.append(ds)
    return out, times


def cyclical_time_features(times: pd.DatetimeIndex):
    doy = times.dayofyear.values.astype(np.float32)
    mon = times.month.values.astype(np.float32)
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    mon_sin = np.sin(2 * np.pi * mon / 12.0).astype(np.float32)
    mon_cos = np.cos(2 * np.pi * mon / 12.0).astype(np.float32)
    return doy_sin, doy_cos, mon_sin, mon_cos


def build_climate_mask(shp_path: str, climate_field: str, ref: xr.Dataset, name_map: dict = None) -> tuple[xr.DataArray, dict]:
    """
    读取 Shapefile 并根据 ref 网格生成 Mask。
    如果提供了 name_map (中文名 -> ID)，则强制使用该 ID。
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS. Please define CRS for Chinese_climate.shp.")
    if gdf.crs.to_string().lower() not in ["epsg:4326", "wgs84"]:
        gdf = gdf.to_crs("EPSG:4326")

    if climate_field not in gdf.columns:
        raise ValueError(f"Field '{climate_field}' not found in shapefile.")

    # 准备 Polygon 和对应的数值 ID
    outlines = []
    numbers = []
    names = [] # 用于记录 ID -> 中文名的映射
    
    # 遍历 Shapefile 的每一行
    for idx, row in gdf.iterrows():
        geom = row.geometry
        c_name = str(row[climate_field]).strip() # 获取中文名
        
        # 如果提供了映射字典，且该名称在字典中
        if name_map and c_name in name_map:
            cid = int(name_map[c_name])
            
            # 如果是 MultiPolygon，RegionMask 也能处理，直接添加
            outlines.append(geom)
            numbers.append(cid)
            names.append(c_name)
        else:
            # 如果没在字典里（比如一些碎小的岛屿或未定义区域），可以选择跳过或报错
            # print(f"Warning: Climate zone '{c_name}' not in map, skipping.")
            pass

    if not outlines:
        raise ValueError("No valid climate zones found matching the provided map!")

    # 创建 RegionMask
    # 注意：regionmask 的 numbers 参数决定了 mask 里的值
    regions = regionmask.Regions(outlines=outlines, numbers=numbers, names=names)
    
    # 生成 Mask (lat, lon)
    # 这里的 mask 值就是我们传入的 numbers (1, 2, 3...)
    mask_da = regions.mask(ref["lon"], ref["lat"])
    
    # regionmask 默认把外部填为 NaN，内部填 ID
    # 我们将其转为 float32 以便后续计算
    climate_id = mask_da.astype("float32")

    # 生成一个反向映射 (ID -> Name) 方便查看
    # 这里的 name_map 是 {中文: ID}，我们需要返回 {ID: 中文}
    id_to_name = {v: k for k, v in name_map.items()}
    
    print(f"Built climate mask with {len(numbers)} regions.")
    return climate_id, id_to_name


def stack_feature_names(precip_names, static_names, dynamic_names):
    # base precip feats
    base = []
    base += [f"pr_{n}" for n in precip_names]  # 3 products
    base += ["pr_mean", "pr_std", "d12", "d13", "d23"]
    base += ["doy_sin", "doy_cos", "mon_sin", "mon_cos"]
    base += static_names
    base += dynamic_names
    base += ["lat", "lon", "climate_id"]
    return base


def median_impute(X: np.ndarray, med: np.ndarray | None = None):
    if med is None:
        med = np.nanmedian(X, axis=0)
    X2 = X.copy()
    inds = np.where(~np.isfinite(X2))
    if inds[0].size > 0:
        X2[inds] = np.take(med, inds[1])
    return X2, med


# -----------------------------
# Data assembly
# -----------------------------
# -----------------------------
# Data assembly (需更新 load_all_inputs)
# -----------------------------
def load_all_inputs(cfg):
    chunks = {"time": 31} # 或读取 cfg["io"]["chunks"]

    precip_var = cfg["data"]["precip_var"]
    ref_var = cfg["data"]["reference_var"]

    # 1. Load Reference (Target Grid)
    print(f"Loading reference: {cfg['data']['reference']['path']}")
    ref_ds = open_ds(cfg["data"]["reference"]["path"], chunks=chunks)
    
    # 检查变量名
    if precip_var not in ref_ds.data_vars and ref_var in ref_ds.data_vars:
        pass # 正常情况
    if ref_var not in ref_ds.data_vars:
        raise ValueError(f"Reference dataset must contain variable '{ref_var}'.")

    # 2. Load Precip Products
    prod_list = cfg["data"]["precip_products"]
    prods = []
    for p in prod_list:
        print(f"Loading product: {p['name']}")
        ds = open_ds(p["path"], chunks=chunks)
        if precip_var not in ds.data_vars:
            raise ValueError(f"{p['name']} does not contain variable '{precip_var}'.")
        # 对齐网格
        ds = align_grid(ds, ref_ds)
        prods.append((p["name"], ds[precip_var]))

    # 3. Load Static Covariates (2D)
    static_covs = []
    for item in cfg["data"].get("static_covariates", []):
        print(f"Loading static: {item['name']}")
        ds = open_ds(item["path"], chunks=None)
        ds = align_grid(ds, ref_ds)
        v = pick_var(ds, item.get("var"), fallback_name=item["name"])
        da = ds[v].load().astype("float32")
        # 确保是 2D (去掉时间维)
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        static_covs.append((item["name"], da))

    # 4. Load Dynamic Covariates (3D)
    dynamic_covs = []
    for item in cfg["data"].get("dynamic_covariates", []):
        print(f"Loading dynamic: {item['name']}")
        ds = open_ds(item["path"], chunks=chunks)
        ds = align_grid(ds, ref_ds)
        v = pick_var(ds, item.get("var"), fallback_name=item["name"])
        dynamic_covs.append((item["name"], ds[v].astype("float32")))

    # 5. Build Climate Mask (Modified)
    print("Building climate mask...")
    # 从配置中获取中文映射字典
    name_map = cfg["region"].get("climate_name_map", None)
    
    climate_id, climate_mapping = build_climate_mask(
        cfg["region"]["china_climate_shp"],
        cfg["region"]["climate_field"],
        ref_ds,
        name_map=name_map  # <--- 关键修改：传入映射字典
    )

    # 6. Time Intersection (对齐所有数据的时间)
    print("Aligning time dimension...")
    ds_list = [ref_ds[[ref_var]]]
    
    # 把所有含时间的数据放入列表求交集
    for _, da in prods:
        ds_list.append(da.to_dataset(name="pr"))
    for _, da in dynamic_covs:
        ds_list.append(da.to_dataset(name="v"))

    # 执行时间求交集
    ds_list_aligned, times = time_intersection(ds_list)
    print(f"Time alignment done. Valid days: {len(times)}")

    # 7. 恢复数据结构
    # 列表顺序: [Ref, Prod1, Prod2..., Dyn1, Dyn2...]
    ref_aligned = ds_list_aligned[0][ref_var]
    
    idx = 1
    prods_aligned = []
    for p in prod_list:
        prods_aligned.append((p["name"], ds_list_aligned[idx]["pr"]))
        idx += 1
        
    dynamic_aligned = []
    for item in cfg["data"].get("dynamic_covariates", []):
        dynamic_aligned.append((item["name"], ds_list_aligned[idx]["v"]))
        idx += 1
        
    # -----------------------------
    # 8. Sensitivity: tile subsampling + mosaic (optional)
    # -----------------------------
    sens = cfg.get("sensitivity", {}) or {}
    if sens.get("enabled", False):
        tile_size = int(sens.get("tile_size", 64))
        tiles = sens.get("tiles", [])
        mosaic_mode = sens.get("mosaic_mode", "hstack")
        override_coords = bool(sens.get("override_coords", True))

        if (not tiles) or (len(tiles) == 0):
            raise ValueError("sensitivity.enabled=True but sensitivity.tiles is empty.")

        def _cut_tile_3d(da, y0, x0):
            return da.isel(lat=slice(y0, y0 + tile_size), lon=slice(x0, x0 + tile_size))

        def _cut_tile_2d(da, y0, x0):
            return da.isel(lat=slice(y0, y0 + tile_size), lon=slice(x0, x0 + tile_size))

        def _mosaic_3d(da3d):
            arr_list = []
            for t in tiles:
                y0, x0 = int(t["y0"]), int(t["x0"])
                blk = _cut_tile_3d(da3d, y0, x0).values  # (T, 64, 64)
                arr_list.append(blk)
            if mosaic_mode == "hstack":
                arr = np.concatenate(arr_list, axis=2)  # (T, 64, 64*N)
            else:
                raise ValueError(f"Unsupported mosaic_mode={mosaic_mode}")
            out = xr.DataArray(arr, dims=("time", "lat", "lon"), coords={"time": da3d["time"].values})
            if override_coords:
                out = out.assign_coords(lat=np.arange(tile_size), lon=np.arange(tile_size * len(tiles)))
            else:
                out = out.assign_coords(lat=_cut_tile_3d(da3d, int(tiles[0]["y0"]), int(tiles[0]["x0"]))["lat"].values,
                                       lon=np.arange(tile_size * len(tiles)))
            return out.astype("float32")

        def _mosaic_2d(da2d):
            arr_list = []
            for t in tiles:
                y0, x0 = int(t["y0"]), int(t["x0"])
                blk = _cut_tile_2d(da2d, y0, x0).values  # (64, 64)
                arr_list.append(blk)
            if mosaic_mode == "hstack":
                arr = np.concatenate(arr_list, axis=1)  # (64, 64*N)
            else:
                raise ValueError(f"Unsupported mosaic_mode={mosaic_mode}")
            out = xr.DataArray(arr, dims=("lat", "lon"))
            if override_coords:
                out = out.assign_coords(lat=np.arange(tile_size), lon=np.arange(tile_size * len(tiles)))
            else:
                out = out.assign_coords(lat=_cut_tile_2d(da2d, int(tiles[0]["y0"]), int(tiles[0]["x0"]))["lat"].values,
                                       lon=np.arange(tile_size * len(tiles)))
            return out.astype("float32")

        # ref_aligned (3D)
        ref_aligned = _mosaic_3d(ref_aligned)

        # products (3D)
        prods_aligned = [(name, _mosaic_3d(da)) for (name, da) in prods_aligned]

        # dynamic covs (3D)
        dynamic_aligned = [(name, _mosaic_3d(da)) for (name, da) in dynamic_aligned]

        # static covs (2D)
        static_covs = [(name, _mosaic_2d(da)) for (name, da) in static_covs]

        # climate_id (2D)
        climate_id = _mosaic_2d(climate_id)

        # 同时把 ref_ds 也做一个“壳”，保证 lat/lon 与拼接后保持一致
        ref_ds = xr.Dataset(coords={"lat": ref_aligned["lat"].values, "lon": ref_aligned["lon"].values, "time": ref_aligned["time"].values})


    return ref_ds, ref_aligned, prods_aligned, static_covs, dynamic_aligned, climate_id, climate_mapping, times


# -----------------------------
# Sampling & feature extraction
# -----------------------------
def sample_indices_by_climate(climate_id_2d: xr.DataArray, samples_per_climate: int, seed: int):
    rng = np.random.default_rng(seed)
    arr = climate_id_2d.values
    valid = np.isfinite(arr)
    if not valid.any():
        raise ValueError("No valid China grid cells found from shapefile mask.")

    # climate ids are 1..K
    ids = np.unique(arr[valid]).astype(int)
    pairs = []

    for cid in ids:
        yy, xx = np.where(arr == cid)
        if yy.size == 0:
            continue
        n = min(samples_per_climate, yy.size)
        sel = rng.choice(yy.size, size=n, replace=False if yy.size >= n else True)
        pairs.append((yy[sel], xx[sel], np.full(n, cid, dtype=np.int16)))

    y_idx = np.concatenate([p[0] for p in pairs])
    x_idx = np.concatenate([p[1] for p in pairs])
    c_idx = np.concatenate([p[2] for p in pairs])
    return y_idx, x_idx, c_idx


def extract_training_matrix(
    ref_pr: xr.DataArray,
    prods: list,
    static_covs: list,
    dynamic_covs: list,
    climate_id: xr.DataArray,
    times: pd.DatetimeIndex,
    cfg
):
    seed = int(cfg["project"].get("random_seed", 42))
    wet_thr = float(cfg["sampling"]["wet_threshold_mmday"])
    batch_size = int(cfg["sampling"].get("batch_size", 200000))

    strategy = cfg["sampling"].get("strategy", "stratified_by_climate")
    if strategy != "stratified_by_climate":
        raise ValueError("This script currently implements 'stratified_by_climate' sampling.")

    spc = int(cfg["sampling"].get("samples_per_climate", 200000))
    y_idx, x_idx, c_idx = sample_indices_by_climate(climate_id, spc, seed=seed)

    # time sampling: for each selected grid cell, sample a random day
    rng = np.random.default_rng(seed + 7)
    n_cells = y_idx.size
    t_idx = rng.integers(0, len(times), size=n_cells, endpoint=False)

    # build features in batches
    precip_names = [n for n, _ in prods]
    static_names = [n for n, _ in static_covs]
    dynamic_names = [n for n, _ in dynamic_covs]
    feat_names = stack_feature_names(precip_names, static_names, dynamic_names)

    X_list = []
    y_list = []

    for i0 in tqdm(range(0, n_cells, batch_size), desc="Build training samples"):
        i1 = min(i0 + batch_size, n_cells)
        t = xr.DataArray(t_idx[i0:i1], dims="n")
        y = xr.DataArray(y_idx[i0:i1], dims="n")
        x = xr.DataArray(x_idx[i0:i1], dims="n")

        # products
        pvals = []
        for _, da in prods:
            pvals.append(da.isel(time=t, lat=y, lon=x).values.astype(np.float32))
        p1, p2, p3 = pvals[0], pvals[1], pvals[2]

        pr_mean = (p1 + p2 + p3) / 3.0
        pr_std = np.std(np.stack([p1, p2, p3], axis=0), axis=0)
        d12 = np.abs(p1 - p2)
        d13 = np.abs(p1 - p3)
        d23 = np.abs(p2 - p3)

        # time cyc features
        tt = times[t_idx[i0:i1]]
        doy_sin, doy_cos, mon_sin, mon_cos = cyclical_time_features(pd.DatetimeIndex(tt))

        # static (2D)
        svals = []
        for _, da2d in static_covs:
            svals.append(da2d.isel(lat=y, lon=x).values.astype(np.float32))

        # dynamic (3D)
        dvals = []
        for _, da3d in dynamic_covs:
            dvals.append(da3d.isel(time=t, lat=y, lon=x).values.astype(np.float32))

        # lat/lon
        latv = ref_pr["lat"].values[y_idx[i0:i1]].astype(np.float32)
        lonv = ref_pr["lon"].values[x_idx[i0:i1]].astype(np.float32)

        climv = c_idx[i0:i1].astype(np.float32)

        # stack feature vector
        feats = [p1, p2, p3, pr_mean, pr_std, d12, d13, d23,
                 doy_sin, doy_cos, mon_sin, mon_cos]
        feats += svals
        feats += dvals
        feats += [latv, lonv, climv]

        Xb = np.stack(feats, axis=1).astype(np.float32)

        # target
        yb = ref_pr.isel(time=t, lat=y, lon=x).values.astype(np.float32)

        X_list.append(Xb)
        y_list.append(yb)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # remove nan target rows
    m = np.isfinite(y)
    X = X[m]
    y = y[m]

    # wet label
    y_wet = (y >= wet_thr).astype(np.int32)

    return X, y, y_wet, feat_names


# -----------------------------
# Train / Predict
# -----------------------------
def train(cfg):
    out_dir = cfg["io"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    ref_ds, ref_pr, prods, static_covs, dynamic_covs, climate_id, climate_mapping, times = load_all_inputs(cfg)

    seed = int(cfg["project"].get("random_seed", 42))
    wet_thr = float(cfg["sampling"]["wet_threshold_mmday"])

    per_clm = cfg["model"].get("per_climate", {})
    enabled = bool(per_clm.get("enabled", False))
    if not enabled:
        raise ValueError("你已要求按气候分区分别训练，请在 JSON 中设置 model.per_climate.enabled=true")

    spc = int(per_clm.get("samples_per_climate", 250000))
    min_n = int(per_clm.get("min_samples_per_climate", 5000))
    prefix = per_clm.get("model_prefix", "Fusion_RF_TwoStage_Climate")

    # climate ids (1..K)
    clim_arr = climate_id.values
    valid = np.isfinite(clim_arr)
    clim_ids = np.unique(clim_arr[valid]).astype(int).tolist()

    clf_cfg = cfg["model"]["classifier"]
    reg_cfg = cfg["model"]["regressor"]
    wet_prob_thr = float(cfg["model"].get("wet_prob_threshold", 0.5))

    models = {}
    train_diags = {}

    for cid in clim_ids:
        # ---- 1) 抽样：只抽该分区格点 ----
        yy, xx = np.where(clim_arr == float(cid))
        if yy.size == 0:
            continue
        rng = np.random.default_rng(seed + cid)
        # 关键：n 直接等于你希望的样本数，而不是被格点数 yy.size 限制
        n = int(spc)
        # 允许格点重复抽样（replace=True），从而可以生成“格点-日”样本
        sel = rng.choice(yy.size, size=n, replace=True)
        y_idx = yy[sel]
        x_idx = xx[sel]
        c_idx = np.full(n, cid, dtype=np.int16)
        
        # 每个格点抽一个随机日（也可以改成多日）
        t_idx = rng.integers(0, len(times), size=n, endpoint=False)
        print(f"[Climate {cid}] n_cells={yy.size}, n_samples={n}")
        # ---- 2) 构建训练矩阵（与原 extract_training_matrix 一致，但这里是单分区）----
        # products
        pvals = []
        t = xr.DataArray(t_idx, dims="n")
        y = xr.DataArray(y_idx, dims="n")
        x = xr.DataArray(x_idx, dims="n")

        for _, da in prods:
            pvals.append(da.isel(time=t, lat=y, lon=x).values.astype(np.float32))
        p1, p2, p3 = pvals[0], pvals[1], pvals[2]

        pr_mean = (p1 + p2 + p3) / 3.0
        pr_std  = np.std(np.stack([p1, p2, p3], axis=0), axis=0)
        d12 = np.abs(p1 - p2)
        d13 = np.abs(p1 - p3)
        d23 = np.abs(p2 - p3)

        tt = pd.DatetimeIndex(times[t_idx])
        doy_sin, doy_cos, mon_sin, mon_cos = cyclical_time_features(tt)

        # static
        svals = []
        for _, da2d in static_covs:
            svals.append(da2d.isel(lat=y, lon=x).values.astype(np.float32))

        # dynamic
        dvals = []
        for _, da3d in dynamic_covs:
            dvals.append(da3d.isel(time=t, lat=y, lon=x).values.astype(np.float32))

        latv = ref_pr["lat"].values[y_idx].astype(np.float32)
        lonv = ref_pr["lon"].values[x_idx].astype(np.float32)
        climv = c_idx.astype(np.float32)

        feats = [p1, p2, p3, pr_mean, pr_std, d12, d13, d23,
                 doy_sin, doy_cos, mon_sin, mon_cos]
        feats += svals
        feats += dvals
        # 注意：per-climate 模型里 climate_id 本质冗余，但保留也无妨
        feats += [latv, lonv, climv]

        X = np.stack(feats, axis=1).astype(np.float32)
        y_true = ref_pr.isel(time=t, lat=y, lon=x).values.astype(np.float32)

        m = np.isfinite(y_true)
        X = X[m]
        y_true = y_true[m]

        if X.shape[0] < min_n:
            print(f"[Skip] climate_id={cid} samples={X.shape[0]} < min_samples_per_climate={min_n}")
            continue

        y_wet = (y_true >= wet_thr).astype(np.int32)

        # ---- 3) 缺失填补 + 训练 ----
        X, med = median_impute(X, med=None)
        y_reg = np.log1p(np.maximum(y_true, 0.0)).astype(np.float32)

        clf = RandomForestClassifier(
            n_estimators=int(clf_cfg.get("n_estimators", 300)),
            min_samples_leaf=int(clf_cfg.get("min_samples_leaf", 2)),
            n_jobs=int(clf_cfg.get("n_jobs", -1)),
            random_state=seed + cid
        )
        clf.fit(X, y_wet)

        reg = RandomForestRegressor(
            n_estimators=int(reg_cfg.get("n_estimators", 500)),
            min_samples_leaf=int(reg_cfg.get("min_samples_leaf", 2)),
            n_jobs=int(reg_cfg.get("n_jobs", -1)),
            random_state=seed + cid
        )
        wet_idx = np.where(y_wet == 1)[0]
        reg.fit(X[wet_idx], y_reg[wet_idx])

        # ---- 4) 快速诊断 ----
        proba = clf.predict_proba(X)[:, 1]
        wet_hat = (proba >= wet_prob_thr)
        y_hat = np.zeros_like(y_true, dtype=np.float32)
        if wet_hat.any():
            y_hat[wet_hat] = np.expm1(reg.predict(X[wet_hat])).astype(np.float32)
        y_hat = np.maximum(y_hat, 0.0)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
        mae  = float(mean_absolute_error(y_true, y_hat))
        r2   = float(r2_score(y_true, y_hat))

        models[str(cid)] = {
            "impute_median": med,
            "clf": clf,
            "reg": reg,
            "diag": {"rmse": rmse, "mae": mae, "r2": r2},
            "climate_name": climate_mapping.get(str(cid), f"climate_{cid}")
        }
        train_diags[str(cid)] = models[str(cid)]["diag"]

        print(f"[Trained] climate_id={cid} name={models[str(cid)]['climate_name']} RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")

    bundle = {
        "wet_threshold_mmday": wet_thr,
        "wet_prob_threshold": float(cfg["model"].get("wet_prob_threshold", 0.5)),
        "climate_mapping": climate_mapping,
        "models": models,
        "train_diags": train_diags
    }

    model_path = os.path.join(out_dir, cfg["io"]["model_file"])
    joblib.dump(bundle, model_path)

    with open(model_path + ".config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n[Train done]")
    print(f"Saved: {model_path}")
    print("Per-climate models:", list(models.keys()))


def predict(cfg):
    out_dir = cfg["io"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ---- load per-climate model bundle ----
    model_path = os.path.join(out_dir, cfg["io"]["model_file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please run train first.")
    bundle = joblib.load(model_path)

    if "models" not in bundle or not isinstance(bundle["models"], dict) or len(bundle["models"]) == 0:
        raise ValueError("Loaded model is not a per-climate bundle (bundle['models'] is empty).")

    models = bundle["models"]
    wet_prob_thr = float(bundle.get("wet_prob_threshold", 0.5))

    # ---- load & align all inputs (reference grid, covariates, climate mask, time intersection) ----
    ref_ds, ref_pr, prods, static_covs, dynamic_covs, climate_id, climate_mapping, times = load_all_inputs(cfg)

    # print output time range (IMPORTANT)
    t0 = pd.to_datetime(times[0]).date()
    t1 = pd.to_datetime(times[-1]).date()
    print(f"[Output time range] {t0} to {t1} (n={len(times)})")

    lat = ref_ds["lat"].values
    lon = ref_ds["lon"].values
    nT = len(times)
    nY = len(lat)
    nX = len(lon)

    # climate_id 2D
    clim2d = climate_id.values.astype(np.float32)

    # cache static 2D
    static2d = {name: da.values.astype(np.float32) for name, da in static_covs}

    # output path
    out_nc = os.path.join(out_dir, cfg["io"]["fused_nc"])
    tmp_dir = out_nc + ".tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    time_batch = int(cfg["io"].get("time_batch_days", 15))
    block_y = int(cfg["io"].get("block_y", 120))
    block_x = int(cfg["io"].get("block_x", 160))
    clevel = int(cfg["io"].get("netcdf_compress_level", 4))

    mask_outside = bool(cfg["region"].get("mask_outside_china_as_nan", True))

    precip_names = [n for n, _ in prods]
    static_names = [n for n, _ in static_covs]
    dynamic_names = [n for n, _ in dynamic_covs]

    part_files = []

    # ---- batch over time ----
    for t_start in tqdm(range(0, nT, time_batch), desc="Predict (time batches)"):
        t_end = min(t_start + time_batch, nT)
        times_blk = pd.DatetimeIndex(times[t_start:t_end])
        T = len(times_blk)

        fused_blk = np.full((T, nY, nX), np.nan, dtype=np.float32)

        # time cyc features
        doy_sin, doy_cos, mon_sin, mon_cos = cyclical_time_features(times_blk)

        # slice dynamic vars for this time block
        dyn_blk = {}
        for name, da in dynamic_covs:
            dyn_blk[name] = da.isel(time=slice(t_start, t_end))

        # slice precip products for this time block
        p_blk = {}
        for name, da in prods:
            p_blk[name] = da.isel(time=slice(t_start, t_end))

        # ---- spatial blocks ----
        for y0 in range(0, nY, block_y):
            y1 = min(y0 + block_y, nY)
            Yb = y1 - y0
            lat_blk_1d = lat[y0:y1].astype(np.float32)

            for x0 in range(0, nX, block_x):
                x1 = min(x0 + block_x, nX)
                Xb = x1 - x0
                lon_blk_1d = lon[x0:x1].astype(np.float32)

                clim_blk = clim2d[y0:y1, x0:x1]  # (Yb, Xb)
                valid_mask_2d = np.isfinite(clim_blk)
                if not valid_mask_2d.any():
                    continue

                # load precip arrays (T, Yb, Xb)
                p1 = p_blk[precip_names[0]].isel(lat=slice(y0, y1), lon=slice(x0, x1)).load().values.astype(np.float32)
                p2 = p_blk[precip_names[1]].isel(lat=slice(y0, y1), lon=slice(x0, x1)).load().values.astype(np.float32)
                p3 = p_blk[precip_names[2]].isel(lat=slice(y0, y1), lon=slice(x0, x1)).load().values.astype(np.float32)

                pr_mean = (p1 + p2 + p3) / 3.0
                pr_std = np.std(np.stack([p1, p2, p3], axis=0), axis=0)
                d12 = np.abs(p1 - p2)
                d13 = np.abs(p1 - p3)
                d23 = np.abs(p2 - p3)

                N = T * Yb * Xb

                # expand time cyc to (T, Yb, Xb)
                doy_sin3 = np.repeat(doy_sin[:, None, None], Yb, axis=1)
                doy_sin3 = np.repeat(doy_sin3, Xb, axis=2)
                doy_cos3 = np.repeat(doy_cos[:, None, None], Yb, axis=1)
                doy_cos3 = np.repeat(doy_cos3, Xb, axis=2)
                mon_sin3 = np.repeat(mon_sin[:, None, None], Yb, axis=1)
                mon_sin3 = np.repeat(mon_sin3, Xb, axis=2)
                mon_cos3 = np.repeat(mon_cos[:, None, None], Yb, axis=1)
                mon_cos3 = np.repeat(mon_cos3, Xb, axis=2)

                # static expand to (T, Yb, Xb)
                s_list = []
                for sn in static_names:
                    s2 = static2d[sn][y0:y1, x0:x1]
                    s3 = np.repeat(s2[None, :, :], T, axis=0)
                    s_list.append(s3.astype(np.float32))

                # dynamic arrays (T, Yb, Xb)
                d_list = []
                for dn in dynamic_names:
                    d3 = dyn_blk[dn].isel(lat=slice(y0, y1), lon=slice(x0, x1)).load().values.astype(np.float32)
                    d_list.append(d3)

                # lat/lon expand to (T, Yb, Xb)
                lat2d = np.repeat(lat_blk_1d[:, None], Xb, axis=1).astype(np.float32)  # (Yb,Xb)
                lon2d = np.repeat(lon_blk_1d[None, :], Yb, axis=0).astype(np.float32)  # (Yb,Xb)
                lat3 = np.repeat(lat2d[None, :, :], T, axis=0)
                lon3 = np.repeat(lon2d[None, :, :], T, axis=0)

                # climate_id expand to (T, Yb, Xb)
                clim3 = np.repeat(clim_blk[None, :, :], T, axis=0).astype(np.float32)

                # assemble features: (F, T, Yb, Xb) -> (N, F)
                feats = [
                    p1, p2, p3,
                    pr_mean, pr_std, d12, d13, d23,
                    doy_sin3, doy_cos3, mon_sin3, mon_cos3
                ]
                feats += s_list
                feats += d_list
                feats += [lat3, lon3, clim3]  # IMPORTANT: last col is climate_id

                Xmat = np.stack(feats, axis=0).astype(np.float32)    # (F, T, Yb, Xb)
                Xmat = Xmat.reshape((Xmat.shape[0], N)).T            # (N, F)

                # climate vector (N,)
                clim_vec = Xmat[:, -1]
                # outside-china rows: clim is NaN; set to 0 for grouping (skip)
                clim_int = np.where(np.isfinite(clim_vec), clim_vec.astype(np.int32), 0)

                y_pred = np.full((N,), np.nan, dtype=np.float32)

                # group predict per climate_id
                uniq_cids = np.unique(clim_int)
                for cid in uniq_cids:
                    if cid <= 0:
                        continue
                    key = str(cid)
                    if key not in models:
                        # 没有该分区模型：保持 NaN
                        continue

                    m = (clim_int == cid)
                    if not m.any():
                        continue

                    # per-climate median impute
                    med = np.array(models[key]["impute_median"], dtype=np.float32)
                    Xm, _ = median_impute(Xmat[m], med=med)

                    clf = models[key]["clf"]
                    reg = models[key]["reg"]

                    proba = clf.predict_proba(Xm)[:, 1].astype(np.float32)
                    wet_hat = (proba >= wet_prob_thr)

                    ym = np.zeros((Xm.shape[0],), dtype=np.float32)
                    if wet_hat.any():
                        ym[wet_hat] = np.expm1(reg.predict(Xm[wet_hat])).astype(np.float32)

                    y_pred[m] = np.maximum(ym, 0.0)

                # reshape back to (T, Yb, Xb)
                y_pred_3d = y_pred.reshape((T, Yb, Xb)).astype(np.float32)

                # apply outside-China mask
                if mask_outside:
                    for tt in range(T):
                        tmp = y_pred_3d[tt]
                        tmp[~valid_mask_2d] = np.nan
                        y_pred_3d[tt] = tmp

                fused_blk[:, y0:y1, x0:x1] = y_pred_3d

        # write part nc
        ds_out = xr.Dataset(
            data_vars={"pr_fused": (("time", "lat", "lon"), fused_blk)},
            coords={"time": times_blk, "lat": lat, "lon": lon}
        )
        ds_out["pr_fused"].attrs.update(
            units="mm/day",
            long_name="Fused daily precipitation (RF Two-stage, per-climate models)"
        )

        part_path = os.path.join(tmp_dir, f"part_{t_start:06d}_{t_end:06d}.nc")
        ds_out.to_netcdf(part_path, encoding={"pr_fused": dict(zlib=True, complevel=clevel, dtype="float32")})
        part_files.append(part_path)

    # merge parts
    ds_all = xr.open_mfdataset(part_files, combine="by_coords")
    ds_all.to_netcdf(out_nc, encoding={"pr_fused": dict(zlib=True, complevel=clevel, dtype="float32")})

    # cleanup
    for f in part_files:
        try:
            os.remove(f)
        except Exception:
            pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    print("\n[Predict done]")
    print(f"Fused NetCDF saved: {out_nc}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train")
    p1.add_argument("--config", required=True)

    p2 = sub.add_parser("predict")
    p2.add_argument("--config", required=True)

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "train":
        train(cfg)
    else:
        predict(cfg)


if __name__ == "__main__":
    main()
