#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][pid={os.getpid()}] {msg}", flush=True)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _norm_unit(u: Optional[str]) -> str:
    return (u or "").strip().lower()


def pr_to_mmday(da: xr.DataArray) -> xr.DataArray:
    u = _norm_unit(da.attrs.get("units"))
    out = da
    if ("kg" in u and "s-1" in u) or ("kg" in u and "/s" in u) or ("kg m-2 s-1" in u):
        out = da * 86400.0
        out.attrs.update(da.attrs)
        out.attrs["units"] = "mm/day"
        return out
    if "mm" in u:
        out.attrs.update(da.attrs)
        out.attrs["units"] = "mm/day"
        return out
    out.attrs.update(da.attrs)
    return out


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for k in ["input_dir", "out_root", "variables", "indices"]:
        if k not in cfg:
            raise SystemExit(f"Missing key in config: {k}")

    cfg.setdefault("recursive", False)
    cfg.setdefault("file_glob", "*.nc")

    cfg.setdefault("product", "PRODUCT")
    cfg.setdefault("scenario", "SCENARIO")

    cfg.setdefault("frequencies", ["year"])
    cfg["frequencies"] = [str(x).lower() for x in cfg["frequencies"]]
    for f0 in cfg["frequencies"]:
        if f0 not in ("year", "month"):
            raise SystemExit(f"Unsupported frequency: {f0} (only year/month)")

    cfg.setdefault("overwrite", False)
    cfg.setdefault("nproc", 1)

    cfg.setdefault("wet_day_thresh_mm", 1.0)

    cfg.setdefault("percentile_base", None)
    cfg.setdefault("percentile_mode", "global")

    cfg.setdefault("varname_map", {})
    cfg.setdefault("auto_detect_pr_var", True)
    cfg.setdefault("pr_var_candidates", ["pr", "precip", "pre", "rain", "precipitation"])

    return cfg


# ---------- percentiles ----------
def _subset_base(da: xr.DataArray, cfg: Dict[str, Any]) -> xr.DataArray:
    base = cfg.get("percentile_base")
    if not base:
        return da
    start = base.get("start")
    end = base.get("end")
    if start and end:
        return da.sel(time=slice(start, end))
    return da


def rXXp_sum(pr_mmday: xr.DataArray, q: float, cfg: Dict[str, Any], wet_thresh: float, freq: str) -> xr.DataArray:
    base = _subset_base(pr_mmday, cfg)
    if base.sizes.get("time", 0) == 0:
        raise ValueError("percentile_base slice produced empty time axis (check start/end and file time range).")
    wet_base = base.where(base >= wet_thresh)
    thr = wet_base.quantile(q, dim="time", skipna=True)
    cond = pr_mmday > thr
    return pr_mmday.where(cond).resample(time=("YS" if freq == "year" else "MS")).sum("time", skipna=True)


# ---------- run-length ----------
def _max_run_1d(arr_bool: np.ndarray) -> np.int32:
    a = np.asarray(arr_bool, dtype=bool)
    if a.size == 0:
        return np.int32(0)
    diff = np.diff(a.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if a[0]:
        starts = np.r_[0, starts]
    if a[-1]:
        ends = np.r_[ends, a.size]
    if starts.size == 0:
        return np.int32(0)
    return np.int32((ends - starts).max())


def r_cdd_cwd(pr_mmday: xr.DataArray, freq: str, wet_thresh: float, which: str) -> xr.DataArray:
    cond = (pr_mmday < wet_thresh) if which == "cdd" else (pr_mmday >= wet_thresh)

    def _maxrun(x):
        return xr.apply_ufunc(
            _max_run_1d, x,
            input_core_dims=[["time"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int32],
        )

    return cond.resample(time=("YS" if freq == "year" else "MS")).map(_maxrun)


# ---------- helpers ----------
def nc_stem(path: str) -> str:
    b = os.path.basename(path)
    return re.sub(r"\.nc$", "", b)


def out_file(out_root: str, product: str, scenario: str, freq: str, idx: str, stem: str) -> str:
    # out_root/product/scenario/freq/index/stem.index.freq.nc
    d = os.path.join(out_root, product, scenario, freq, idx)
    ensure_dir(d)
    return os.path.join(d, f"{stem}.{idx}.{freq}.nc")


def write_one(path: str, idx: str, da: xr.DataArray):
    da2 = da.astype("float32")
    da2.name = idx
    ds = da2.to_dataset()
    enc = {idx: {"zlib": True, "complevel": 4}}
    tmp = path + ".tmp"
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)


def normalize_pr_var(ds: xr.Dataset, cfg: Dict[str, Any]) -> xr.Dataset:
    # 1) 显式映射优先
    mp = cfg.get("varname_map") or {}
    src = mp.get("pr")
    if src and (src in ds.data_vars) and ("pr" not in ds.data_vars):
        return ds.rename({src: "pr"})

    # 2) 自动探测
    if cfg.get("auto_detect_pr_var", True) and ("pr" not in ds.data_vars):
        for cand in cfg.get("pr_var_candidates", []):
            if cand in ds.data_vars:
                log(f"[INFO] auto-detect pr var: {cand} -> pr")
                return ds.rename({cand: "pr"})

    return ds


def compute_pr_indices(ds: xr.Dataset, cfg: Dict[str, Any], freq: str) -> Dict[str, xr.DataArray]:
    wet_thresh = float(cfg.get("wet_day_thresh_mm", 1.0))
    pr = pr_to_mmday(ds["pr"])
    rule = "YS" if freq == "year" else "MS"

    out = {}
    for idx in cfg["indices"].get("pr", []):
        if idx == "rx1day":
            out[idx] = pr.resample(time=rule).max("time", skipna=True)
        elif idx == "rx5day":
            def _rx5(x):
                r = x.rolling(time=5, min_periods=5).sum()
                return r.max("time", skipna=True)
            out[idx] = pr.resample(time=rule).map(_rx5)
        elif idx == "prcptot":
            wet = pr >= wet_thresh
            out[idx] = pr.where(wet).resample(time=rule).sum("time", skipna=True)
        elif idx == "sdii":
            wet = pr >= wet_thresh
            tot = pr.where(wet).resample(time=rule).sum("time", skipna=True)
            nwet = wet.resample(time=rule).sum("time", skipna=True)
            out[idx] = tot / xr.where(nwet == 0, np.nan, nwet)
        elif idx == "cdd":
            out[idx] = r_cdd_cwd(pr, freq, wet_thresh, which="cdd")
        elif idx == "cwd":
            out[idx] = r_cdd_cwd(pr, freq, wet_thresh, which="cwd")
        elif idx == "r10mm":
            out[idx] = (pr >= 10.0).resample(time=rule).sum("time", skipna=True)
        elif idx == "r20mm":
            out[idx] = (pr >= 20.0).resample(time=rule).sum("time", skipna=True)
        elif idx == "r95p":
            out[idx] = rXXp_sum(pr, 0.95, cfg, wet_thresh, freq)
        elif idx == "r99p":
            out[idx] = rXXp_sum(pr, 0.99, cfg, wet_thresh, freq)
        else:
            raise ValueError(f"Unknown pr index: {idx}")

    return out


def discover_files(cfg: Dict[str, Any]) -> List[str]:
    root = cfg["input_dir"]
    patt = cfg.get("file_glob", "*.nc")
    recursive = bool(cfg.get("recursive", False))

    if recursive:
        files = glob.glob(os.path.join(root, "**", patt), recursive=True)
    else:
        files = glob.glob(os.path.join(root, patt), recursive=False)

    files = sorted([f for f in files if os.path.isfile(f)])
    return files


def process_one_file(args):
    cfg, in_nc = args
    stem = nc_stem(in_nc)

    overwrite = bool(cfg.get("overwrite", False))
    out_root = cfg["out_root"]
    product = cfg.get("product", "PRODUCT")
    scenario = cfg.get("scenario", "SCENARIO")

    try:
        ds = xr.open_dataset(in_nc, decode_times=True)
    except Exception as e:
        log(f"[WARN] open failed: {in_nc} | {e}")
        return

    # 仅处理 pr（如果你要扩展到温度，我可以继续把 tasmin/tasmax/pair 接进来）
    if "pr" in cfg["variables"]:
        ds = normalize_pr_var(ds, cfg)
        if "pr" not in ds.data_vars:
            log(f"[WARN] skip (no pr var): {in_nc} | vars={list(ds.data_vars)}")
            ds.close()
            return

    for freq in cfg["frequencies"]:
        try:
            results = compute_pr_indices(ds, cfg, freq=freq)
        except Exception as e:
            log(f"[WARN] compute failed: {in_nc} | freq={freq} | {e}")
            continue

        for idx, da in results.items():
            outp = out_file(out_root, product, scenario, freq, idx, stem)

            if (not overwrite) and os.path.exists(outp) and os.path.getsize(outp) > 0:
                # 已完成就跳过
                continue

            try:
                write_one(outp, idx, da)
            except Exception as e:
                log(f"[WARN] write failed: {outp} | {e}")

    ds.close()
    log(f"[DONE] {in_nc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    files = discover_files(cfg)
    log(f"Found {len(files)} files under {cfg['input_dir']}")

    if not files:
        return

    nproc = int(cfg.get("nproc", 1))
    tasks = [(cfg, f) for f in files]

    if nproc <= 1:
        for t in tasks:
            process_one_file(t)
    else:
        log(f"Run with nproc={nproc}")
        with Pool(processes=nproc) as pool:
            pool.map(process_one_file, tasks)

    log("ALL DONE")


if __name__ == "__main__":
    main()
