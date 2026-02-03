#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# Variable / coord helpers
# -----------------------------
def pick_var(ds: xr.Dataset, var_hint: Optional[str] = None) -> str:
    if var_hint and var_hint in ds.data_vars:
        return var_hint
    candidates = ["pr", "prec", "PRE", "precip", "precipitation", "ppt", "rain", "P", "tp",
                  "elev", "elevation", "z", "topo", "dem", "orog", "height"]
    for c in candidates:
        if c in ds.data_vars:
            return c
    return list(ds.data_vars.keys())[0]

def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    # common lon/lat naming
    for a, b in [("longitude", "lon"), ("latitude", "lat"),
                 ("Longitude", "lon"), ("Latitude", "lat"),
                 ("LONGITUDE", "lon"), ("LATITUDE", "lat")]:
        if a in ds.coords and b not in ds.coords:
            rename[a] = b
        if a in ds.dims and b not in ds.dims:
            rename[a] = b

    # DEM-style x/y naming (your ETOPO2 case)
    if "x" in ds.coords and "lon" not in ds.coords:
        rename["x"] = "lon"
    if "y" in ds.coords and "lat" not in ds.coords:
        rename["y"] = "lat"
    if "x" in ds.dims and "lon" not in ds.dims:
        rename["x"] = "lon"
    if "y" in ds.dims and "lat" not in ds.dims:
        rename["y"] = "lat"

    if rename:
        ds = ds.rename(rename)
    return ds


def ensure_mm_per_day(da: xr.DataArray) -> xr.DataArray:
    units = (da.attrs.get("units", "") or "").lower().strip()
    if "kg" in units and "s-1" in units:
        da = da * 86400.0
        da.attrs["units"] = "mm/day (converted from kg m-2 s-1)"
    elif units in ["m", "meter", "metre"] or units.endswith(" m"):
        da = da * 1000.0
        da.attrs["units"] = "mm/day (converted from m)"
    return da

def align_to_ref(ref: xr.DataArray, x: xr.DataArray, regrid: str = "none") -> Tuple[xr.DataArray, xr.DataArray]:
    ref2, x2 = xr.align(ref, x, join="inner")

    # grid check + optional interp
    if all(c in ref2.coords for c in ["lat", "lon"]) and all(c in x2.coords for c in ["lat", "lon"]):
        same_lat = (ref2.lat.size == x2.lat.size) and np.allclose(ref2.lat.values, x2.lat.values)
        same_lon = (ref2.lon.size == x2.lon.size) and np.allclose(ref2.lon.values, x2.lon.values)
        if (not same_lat) or (not same_lon):
            if regrid == "none":
                raise ValueError("lat/lon grid mismatch. Set regrid=nearest or linear in config.")
            method = "nearest" if regrid == "nearest" else "linear"
            x2 = x2.interp(lat=ref2.lat, lon=ref2.lon, method=method)
    return ref2, x2

def mask_valid(ref: xr.DataArray, x: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    m = np.isfinite(ref) & np.isfinite(x)
    return ref.where(m), x.where(m)

# -----------------------------
# Metrics: per-grid over time
# -----------------------------
def _m(a, dim="time"): return a.mean(dim=dim, skipna=True)
def _s(a, dim="time"): return a.std(dim=dim, skipna=True)

def bias(ref, x): return _m(x - ref)
def mae(ref, x): return _m(np.abs(x - ref))
def rmse(ref, x): return np.sqrt(_m((x - ref) ** 2))

def corr(ref, x):
    ra = ref - _m(ref)
    xa = x - _m(x)
    num = _m(ra * xa)
    den = _s(ref) * _s(x)
    return num / den

def kge(ref, x):
    r = corr(ref, x)
    mean_ref = _m(ref)
    mean_x = _m(x)
    std_ref = _s(ref)
    std_x = _s(x)
    alpha = std_x / std_ref
    beta = mean_x / mean_ref
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

def summary_global(ref, x) -> Dict[str, float]:
    r = xr.corr(ref, x, dim=("time", "lat", "lon"))
    out = {
        "Bias": float((x - ref).mean(skipna=True).values),
        "MAE": float(np.abs(x - ref).mean(skipna=True).values),
        "RMSE": float(np.sqrt(((x - ref) ** 2).mean(skipna=True)).values),
        "Corr": float(r.values),
    }
    mean_ref = float(ref.mean(skipna=True).values)
    mean_x = float(x.mean(skipna=True).values)
    std_ref = float(ref.std(skipna=True).values)
    std_x = float(x.std(skipna=True).values)
    alpha = std_x / std_ref if std_ref != 0 else np.nan
    beta = mean_x / mean_ref if mean_ref != 0 else np.nan
    out["KGE"] = float(1.0 - np.sqrt((out["Corr"] - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return out

# -----------------------------
# Event metrics
# -----------------------------
def event_metrics(ref: xr.DataArray, x: xr.DataArray, thr: float) -> xr.Dataset:
    obs = (ref >= thr)
    sim = (x >= thr)

    H = (sim & obs).sum("time")
    M = ((~sim) & obs).sum("time")
    F = (sim & (~obs)).sum("time")
    CN = ((~sim) & (~obs)).sum("time")

    POD = H / (H + M)
    FAR = F / (H + F)
    CSI = H / (H + M + F)
    HSS = (2*(H*CN - M*F)) / ((H+M)*(M+CN) + (H+F)*(F+CN))

    ds = xr.Dataset(
        data_vars=dict(hits=H, misses=M, false_alarms=F, correct_negatives=CN,
                       POD=POD, FAR=FAR, CSI=CSI, HSS=HSS)
    )
    ds.attrs["event_threshold_mm_day"] = float(thr)
    return ds

# -----------------------------
# Climate-zone mask from shapefile
# -----------------------------
def build_zone_mask(lat: xr.DataArray, lon: xr.DataArray, shp_path: str, field: str) -> Tuple[xr.DataArray, List[str]]:
    try:
        import geopandas as gpd
        import regionmask
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for zonal mask. Install:\n"
            "  conda install -c conda-forge geopandas regionmask\n"
            f"Original error: {e}"
        )

    gdf = gpd.read_file(shp_path)
    if field not in gdf.columns:
        raise ValueError(f"Field '{field}' not found in shapefile. Available: {list(gdf.columns)}")

    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")

    zone_names = [str(v) for v in gdf[field].values]
    regs = regionmask.Regions(outlines=list(gdf.geometry), names=zone_names, numbers=list(range(len(zone_names))))
    zone_id = regs.mask(lon, lat)  # (lat, lon), NaN outside
    zone_id = zone_id.fillna(-1).astype("int16")
    zone_id.name = "zone_id"
    zone_id.attrs["shp_path"] = shp_path
    zone_id.attrs["field"] = field
    return zone_id, zone_names

def zonal_mean(da: xr.DataArray, zone_id: xr.DataArray, zone_index: int) -> xr.DataArray:
    m = xr.where(zone_id == zone_index, 1.0, np.nan)
    mm = m.broadcast_like(da.isel(time=0) if "time" in da.dims else da)
    return da.where(np.isfinite(mm)).mean(("lat", "lon"), skipna=True)

# -----------------------------
# Elevation bands
# -----------------------------
def load_dem_on_ref_grid(ref: xr.DataArray, dem_path: str, dem_var_hint: Optional[str], regrid: str, land_only: bool) -> xr.DataArray:
    dem_ds = xr.open_dataset(dem_path)  # xarray 默认会 decode scale_factor/add_offset/_FillValue
    dem_ds = standardize_latlon(dem_ds)

    dv = pick_var(dem_ds, dem_var_hint)
    dem = dem_ds[dv].astype("float32")

    # 可选：仅保留陆地区域海拔（>0m）；海洋与非有限值置为 NaN
    if land_only:
        dem = dem.where(dem > 0)

    # DEM 对齐到 ref 网格（DEM 通常分辨率不同，必须插值）
    if "lat" in dem.coords and "lon" in dem.coords:
        # 即便 run.regrid=none，也建议 DEM 至少用 nearest 对齐
        dem_regrid_method = "nearest" if regrid == "none" else regrid
        _, dem2 = align_to_ref(ref.isel(time=0), dem, regrid=dem_regrid_method)
        dem2.name = "dem"
        dem2.attrs["source_dem"] = dem_path
        dem2.attrs["source_var"] = dv
        dem2.attrs["land_only"] = str(land_only)
        return dem2
    else:
        raise ValueError("DEM does not have lat/lon coords after standardization (expect x/y or lon/lat).")


def build_elev_band_id(dem: xr.DataArray, bands: List[float]) -> Tuple[xr.DataArray, List[str]]:
    """
    bands: e.g. [0,500,1000,2000,3000,4000,9000]
    Returns band_id (lat,lon) int16 with -1 outside/NaN; band_names
    """
    bands = [float(b) for b in bands]
    if len(bands) < 2:
        raise ValueError("bands_m must have at least 2 edges.")
    band_names = []
    band_id = xr.full_like(dem, fill_value=-1, dtype="int16")

    for i in range(len(bands) - 1):
        lo, hi = bands[i], bands[i+1]
        if i == 0:
            mask = (dem >= lo) & (dem < hi)
            label = f"[{int(lo)},{int(hi)})m"
        else:
            mask = (dem >= lo) & (dem < hi)
            label = f"[{int(lo)},{int(hi)})m"
        band_id = xr.where(mask, i, band_id)
        band_names.append(label)

    band_id = band_id.where(np.isfinite(dem), other=-1).astype("int16")
    band_id.name = "elev_band_id"
    band_id.attrs["bands_m"] = bands
    return band_id, band_names

def band_mean(da: xr.DataArray, band_id: xr.DataArray, band_index: int) -> xr.DataArray:
    m = xr.where(band_id == band_index, 1.0, np.nan)
    mm = m.broadcast_like(da.isel(time=0) if "time" in da.dims else da)
    return da.where(np.isfinite(mm)).mean(("lat", "lon"), skipna=True)

# -----------------------------
# Quick plots (optional)
# -----------------------------
def quick_plots(out_dir: str, ref: xr.DataArray, targets: Dict[str, xr.DataArray]):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    ref_ts = ref.mean(("lat", "lon"), skipna=True)
    plt.figure()
    plt.plot(ref_ts["time"].values, ref_ts.values, label="REF")
    for name, da in targets.items():
        ts = da.mean(("lat", "lon"), skipna=True)
        plt.plot(ts["time"].values, ts.values, label=name, alpha=0.8)
    plt.legend()
    plt.title("China mean daily precipitation")
    plt.xlabel("Time")
    plt.ylabel("Precip (mm/day)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "timeseries_china_mean.png"), dpi=600)
    plt.close()

    for name, da in targets.items():
        rm = rmse(ref, da)
        plt.figure()
        rm.plot()
        plt.title(f"RMSE vs REF: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rmse_map_{name}.png"), dpi=600)
        plt.close()

# -----------------------------
# Run
# -----------------------------
def run_from_config(cfg: dict):
    ref_path = cfg["ref"]["path"]
    ref_var_hint = cfg["ref"].get("var", None)

    targets_spec = cfg["targets"]
    shp_cfg = cfg.get("china_climate_shp", None)
    dem_cfg = cfg.get("dem", None)
    run_cfg = cfg.get("run", {})

    out_dir = run_cfg["out_dir"]
    regrid = run_cfg.get("regrid", "none")
    wet_thr = float(run_cfg.get("wet_thr", 0.1))
    heavy_thr = float(run_cfg.get("heavy_thr", 10.0))
    chunks = run_cfg.get("chunks", {"time": 3650})

    os.makedirs(out_dir, exist_ok=True)

    # open ref
    ref_ds = xr.open_dataset(ref_path, chunks=chunks)
    ref_ds = standardize_latlon(ref_ds)
    ref_var = pick_var(ref_ds, ref_var_hint)
    ref = ensure_mm_per_day(ref_ds[ref_var]).astype("float32")

    # climate zone mask (optional)
    zone_id, zone_names = None, None
    if shp_cfg is not None:
        zone_id, zone_names = build_zone_mask(ref["lat"], ref["lon"], shp_cfg["path"], shp_cfg["field"])
        zone_id.to_netcdf(os.path.join(out_dir, "china_climate_zone_id.nc"))

    # elevation band mask (optional)
    elev_band_id, elev_band_names, dem_on_ref = None, None, None
    if dem_cfg is not None:
        land_only = bool(dem_cfg.get("land_only", False))
        dem_on_ref = load_dem_on_ref_grid(ref, dem_cfg["path"], dem_cfg.get("var", None), regrid=regrid, land_only=land_only)

        elev_band_id, elev_band_names = build_elev_band_id(dem_on_ref, dem_cfg.get("bands_m", [0,500,1000,2000,3000,4000,9000]))
        dem_on_ref.rename("dem").to_netcdf(os.path.join(out_dir, "dem_on_ref_grid.nc"))
        elev_band_id.to_netcdf(os.path.join(out_dir, "elev_band_id.nc"))

    results_global = []
    results_zone = []
    results_elev = []
    metrics_maps = {}
    event_maps = {}
    targets_data = {}

    season_map = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}

    for item in targets_spec:
        name = item["name"]
        path = item["path"]
        var_hint = item.get("var", None)

        ds = xr.open_dataset(path, chunks=chunks)
        ds = standardize_latlon(ds)
        v = pick_var(ds, var_hint)
        x = ensure_mm_per_day(ds[v]).astype("float32")

        # align & mask
        ref_a, x_a = align_to_ref(ref, x, regrid=regrid)
        ref_m, x_m = mask_valid(ref_a, x_a)

        targets_data[name] = x_m

        # continuous metrics maps (per grid)
        cont = xr.Dataset(
            dict(
                Bias=bias(ref_m, x_m),
                MAE=mae(ref_m, x_m),
                RMSE=rmse(ref_m, x_m),
                Corr=corr(ref_m, x_m),
                KGE=kge(ref_m, x_m),
            )
        )
        cont.attrs.update({"target_name": name, "target_path": path, "target_var": v})
        metrics_maps[name] = cont

        # event maps
        ev_wet = event_metrics(ref_m, x_m, wet_thr)
        ev_heavy = event_metrics(ref_m, x_m, heavy_thr)
        ev = xr.Dataset(
            dict(
                wet_POD=ev_wet["POD"], wet_FAR=ev_wet["FAR"], wet_CSI=ev_wet["CSI"], wet_HSS=ev_wet["HSS"],
                heavy_POD=ev_heavy["POD"], heavy_FAR=ev_heavy["FAR"], heavy_CSI=ev_heavy["CSI"], heavy_HSS=ev_heavy["HSS"],
            )
        )
        ev.attrs.update({"target_name": name, "wet_thr_mm_day": wet_thr, "heavy_thr_mm_day": heavy_thr})
        event_maps[name] = ev

        # global summary: ALL + seasons
        g_all = summary_global(ref_m, x_m)
        g_all.update({"Target": name, "Season": "ALL", "Var": v, "Path": path})
        results_global.append(g_all)

        month = ref_m["time"].dt.month
        for s, ms in season_map.items():
            idx = month.isin(ms)
            g_s = summary_global(ref_m.where(idx), x_m.where(idx))
            g_s.update({"Target": name, "Season": s, "Var": v, "Path": path})
            results_global.append(g_s)

        # climate zones
        if zone_id is not None:
            for zi, zname in enumerate(zone_names):
                ref_z = zonal_mean(ref_m, zone_id, zi)
                x_z = zonal_mean(x_m, zone_id, zi)

                # pack as pseudo-grid to reuse summary_global
                rz = ref_z.expand_dims(lat=[0], lon=[0])
                xz = x_z.expand_dims(lat=[0], lon=[0])

                row = summary_global(rz, xz)
                row.update({"Target": name, "Zone": zname, "Season": "ALL"})
                results_zone.append(row)

                for s, ms in season_map.items():
                    idx = month.isin(ms)
                    rz_s = ref_z.where(idx).expand_dims(lat=[0], lon=[0])
                    xz_s = x_z.where(idx).expand_dims(lat=[0], lon=[0])
                    row_s = summary_global(rz_s, xz_s)
                    row_s.update({"Target": name, "Zone": zname, "Season": s})
                    results_zone.append(row_s)

        # elevation bands
        if elev_band_id is not None:
            for bi, bname in enumerate(elev_band_names):
                ref_b = band_mean(ref_m, elev_band_id, bi)
                x_b = band_mean(x_m, elev_band_id, bi)

                rb = ref_b.expand_dims(lat=[0], lon=[0])
                xb = x_b.expand_dims(lat=[0], lon=[0])

                row = summary_global(rb, xb)
                row.update({"Target": name, "ElevBand": bname, "Season": "ALL"})
                results_elev.append(row)

                for s, ms in season_map.items():
                    idx = month.isin(ms)
                    rb_s = ref_b.where(idx).expand_dims(lat=[0], lon=[0])
                    xb_s = x_b.where(idx).expand_dims(lat=[0], lon=[0])
                    row_s = summary_global(rb_s, xb_s)
                    row_s.update({"Target": name, "ElevBand": bname, "Season": s})
                    results_elev.append(row_s)

        print(f"[OK] {name}: time={ref_m.sizes.get('time', 'NA')} var={v}")

    # write outputs
    for name, dset in metrics_maps.items():
        dset.to_netcdf(os.path.join(out_dir, f"metrics_continuous_{name}.nc"))
    for name, dset in event_maps.items():
        dset.to_netcdf(os.path.join(out_dir, f"metrics_event_{name}.nc"))

    df_global = pd.DataFrame(results_global)
    df_global = df_global[["Target", "Season", "Bias", "MAE", "RMSE", "Corr", "KGE", "Var", "Path"]]
    df_global.to_csv(os.path.join(out_dir, "summary_global_and_seasonal.csv"), index=False)

    if results_zone:
        df_zone = pd.DataFrame(results_zone)
        df_zone = df_zone[["Target", "Zone", "Season", "Bias", "MAE", "RMSE", "Corr", "KGE"]]
        df_zone.to_csv(os.path.join(out_dir, "summary_by_climate_zone.csv"), index=False)

    if results_elev:
        df_elev = pd.DataFrame(results_elev)
        df_elev = df_elev[["Target", "ElevBand", "Season", "Bias", "MAE", "RMSE", "Corr", "KGE"]]
        df_elev.to_csv(os.path.join(out_dir, "summary_by_elevation_band.csv"), index=False)

    quick_plots(os.path.join(out_dir, "plots_quick"), ref, targets_data)

    print("\n[DONE] Written to:", out_dir)
    print(" - summary_global_and_seasonal.csv")
    if results_zone:
        print(" - summary_by_climate_zone.csv + china_climate_zone_id.nc")
    if results_elev:
        print(" - summary_by_elevation_band.csv + dem_on_ref_grid.nc + elev_band_id.nc")
    print(" - metrics_continuous_*.nc")
    print(" - metrics_event_*.nc")
    print(" - plots_quick/*.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_from_config(cfg)

if __name__ == "__main__":
    main()
