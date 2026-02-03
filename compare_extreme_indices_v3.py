# -*- coding: utf-8 -*-
"""
Compare extreme precipitation indices across 8 products (month or year):
- 8×N score heatmap (RMSE/MAE/BIAS/ABIAS/CORR vs REF)
- Significance tests (optional) + FDR q-values (optional)
- Chain analysis: RAW→BC and BC→FUSED as improvement heatmaps (recommended)
- Per-pair grouped bars: RAW vs BC vs FUSED (clearer than many lines)

Path format (your structure):
ROOT/<PRODUCT>/<FREQ>/pr/<INDEX>/<PRODUCT>.<INDEX>.<FREQ>.nc

Run directly in PyCharm (no argparse). Edit PARAMETERS below.
"""

import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =========================
# PARAMETERS (EDIT HERE)
# =========================
ROOT = r"/home/ud202380664/PRE-FUSION/icclim_ET_OUT_correct"
OUT_DIR = r"/home/ud202380664/PRE-FUSION/icclim_ET_OUT_correct/_COMPARE_OUT_V3"

# Choose frequency: "month" or "year"
FREQ = "month"
VAR = "pr"

# Products (8 groups)
PRODUCTS = [
    "China_Fused_Final_CNN",
    "CMFDV2",
    "CPC",
    "MSWEP-V315",
    "MSWX-V100",
    "CPC-BC",
    "MSWEP-V315-BC",
    "MSWX-V100-BC",
]

# Reference product name
REF_PRODUCT = "CMFDV2"

# Indices (N)
INDICES = ["rx1day", "rx5day", "sdii", "cdd", "cwd", "r10mm", "r20mm", "r95p", "r99p", "prcptot"]

# Score type for main heatmap (vs REF):
#   "rmse": lower better
#   "mae" : lower better
#   "bias": signed mean(prod-ref) over spatial cells (can be +/-)
#   "abias": abs(mean(prod-ref)) lower better
#   "corr": spatial Pearson corr, higher better
SCORE_TYPE = "rmse"

# Normalize RMSE/MAE/ABIAS by |REF mean| (dimensionless). Recommended True, but can amplify when REF mean small.
NORMALIZE_BY_REF_MEAN = True

# Time subset (optional)
TIME_START = None  # "2000-01-01"
TIME_END   = None  # "2022-12-31"

# Mask by shapefile (optional; requires geopandas+regionmask). If missing deps -> auto disables.
USE_SHP_MASK = False
SHP_PATH = r"/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp"
SHP_NAME_FIELD = "climate"
SHP_NAME_VALUE = None

# Output controls
DPI = 600
MAKE_MAIN_HEATMAP = True
SIGNIFICANCE_TESTS = True          # requires scipy; if missing -> auto disables
PAIRED_TEST = "wilcoxon"           # "wilcoxon" or "ttest"
FDR_ALPHA = 0.05

# Heatmap styling (readability)
HEATMAP_CMAP = "cividis"           # "cividis" / "magma" / "viridis"
HEATMAP_VMAX_PCTL = 95             # percentile clipping for vmax
HEATMAP_VMIN = 0.0                 # for rmse/mae/abias; for bias/corr we'll auto set
HEATMAP_ANNOTATE = True
HEATMAP_ANNOTATE_STARS = True      # stars from q-values (FDR)
HEATMAP_TEXT_OUTLINE = True
HEATMAP_TEXT_BBOX = False          # if still hard to read, set True

# Chain analysis settings
DO_CHAIN_IMPROVEMENT_HEATMAPS = True
DO_CHAIN_BAR_PLOTS = True

RAW_BC_PAIRS = [
    ("CPC", "CPC-BC"),
    ("MSWEP-V315", "MSWEP-V315-BC"),
    ("MSWX-V100", "MSWX-V100-BC"),
]
FUSED_NAME = "China_Fused_Final_CNN"


# =========================
# Optional deps
# =========================
try:
    import geopandas as gpd
    import regionmask
except Exception:
    USE_SHP_MASK = False

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    SIGNIFICANCE_TESTS = False


# =========================
# Helpers
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def build_fp(product, index):
    # ROOT/<PRODUCT>/<FREQ>/<VAR>/<INDEX>/<PRODUCT>.<INDEX>.<FREQ>.nc
    return os.path.join(ROOT, product, FREQ, VAR, index, f"{product}.{index}.{FREQ}.nc")

def subset_time(da: xr.DataArray):
    if "time" not in da.dims:
        return da
    if TIME_START is None and TIME_END is None:
        return da
    t0 = TIME_START if TIME_START is not None else str(da["time"].values[0])[:10]
    t1 = TIME_END if TIME_END is not None else str(da["time"].values[-1])[:10]
    return da.sel(time=slice(t0, t1))

def to_latlon_names(da: xr.DataArray):
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]
    lat = next((k for k in lat_candidates if k in da.coords or k in da.dims), None)
    lon = next((k for k in lon_candidates if k in da.coords or k in da.dims), None)
    return lat, lon

def pick_data_var(ds: xr.Dataset, index_name: str = None):
    """
    Robust selection of main variable:
    - exact match case-insensitive: CDD vs cdd
    - contains match: CDD_month, prcptot_ann, etc.
    - otherwise choose largest data var (by size), skipping typical auxiliary vars
    """
    if ds is None or len(ds.data_vars) == 0:
        raise ValueError("No data variable found in nc.")

    drop_names = set([
        "crs", "spatial_ref", "time_bnds", "time_bounds", "lat_bnds", "lon_bnds",
        "bounds", "bnds", "height", "orog"
    ])
    candidates = [v for v in ds.data_vars if v.lower() not in drop_names]
    if len(candidates) == 0:
        candidates = list(ds.data_vars)

    if index_name:
        idx = str(index_name).strip().lower()
        for v in candidates:
            if v.lower() == idx:
                return v
        for v in candidates:
            if idx in v.lower():
                return v

        # common fallback names
        fallback = ["index", "indicator", "value"]
        for name in fallback:
            for v in candidates:
                if v.lower() == name:
                    return v

    def var_size(v):
        try:
            return int(np.prod(ds[v].shape))
        except Exception:
            return -1

    candidates_sorted = sorted(candidates, key=var_size, reverse=True)
    return candidates_sorted[0]

def build_region_mask(da: xr.DataArray):
    if (not USE_SHP_MASK) or (SHP_PATH is None) or (not os.path.exists(SHP_PATH)):
        return None

    gdf = gpd.read_file(SHP_PATH)
    if SHP_NAME_FIELD and SHP_NAME_VALUE is not None:
        gdf = gdf[gdf[SHP_NAME_FIELD].astype(str) == str(SHP_NAME_VALUE)]
        if len(gdf) == 0:
            raise ValueError(f"No geometry matched {SHP_NAME_FIELD} == {SHP_NAME_VALUE}")
    if gdf.crs is not None and "4326" not in str(gdf.crs):
        gdf = gdf.to_crs("EPSG:4326")

    lat_name, lon_name = to_latlon_names(da)
    if lat_name is None or lon_name is None:
        raise ValueError("Cannot find lat/lon in data array coords/dims.")

    lats = da[lat_name].values
    lons = da[lon_name].values

    regions = regionmask.Regions(outlines=list(gdf.geometry), names=["REGION"], abbrevs=["R"])
    mask = regions.mask(lons, lats)  # (lat, lon)
    inside = np.isfinite(mask.values)
    return xr.DataArray(inside, coords={lat_name: da[lat_name], lon_name: da[lon_name]}, dims=(lat_name, lon_name))

def climatology_field(da: xr.DataArray, region_mask=None) -> xr.DataArray:
    da2 = subset_time(da)
    if "time" in da2.dims:
        da2 = da2.mean("time", skipna=True)
    lat, lon = to_latlon_names(da2)
    if region_mask is not None and lat in region_mask.dims and lon in region_mask.dims:
        da2 = da2.where(region_mask)
    return da2

def flatten_pair(a: xr.DataArray, b: xr.DataArray):
    aa, bb = xr.align(a, b, join="inner")
    va = aa.values.astype(float).ravel()
    vb = bb.values.astype(float).ravel()
    m = np.isfinite(va) & np.isfinite(vb)
    return va[m], vb[m]

def better_is_lower(score_type):
    return score_type in ["rmse", "mae", "abias"]

def allow_normalize(score_type):
    # normalization by REF mean makes sense mainly for magnitude-type metrics
    return score_type in ["rmse", "mae", "abias"]

def normalize_score(score, ref_mean, score_type):
    if (not NORMALIZE_BY_REF_MEAN) or (not allow_normalize(score_type)):
        return score
    if not np.isfinite(score) or not np.isfinite(ref_mean) or abs(ref_mean) < 1e-12:
        return score
    return score / abs(ref_mean)


def spatial_score(prod_field: xr.DataArray, ref_field: xr.DataArray, score_type: str):
    vp, vr = flatten_pair(prod_field, ref_field)
    if vp.size == 0:
        return np.nan

    diff = vp - vr

    if score_type == "rmse":
        return float(np.sqrt(np.mean(diff**2)))
    if score_type == "mae":
        return float(np.mean(np.abs(diff)))
    if score_type == "bias":
        return float(np.mean(diff))
    if score_type == "abias":
        return float(np.abs(np.mean(diff)))
    if score_type == "corr":
        if vp.size < 3:
            return np.nan
        return float(np.corrcoef(vp, vr)[0, 1])

    if score_type == "nse":
        # NSE = 1 - SSE/SST
        sse = float(np.sum((vp - vr) ** 2))
        sst = float(np.sum((vr - np.mean(vr)) ** 2))
        if sst < 1e-12:   # REF几乎常数 -> NSE不可定义
            return np.nan
        return float(1.0 - sse / sst)

    raise ValueError(f"Unknown SCORE_TYPE: {score_type}")


def paired_pvalue(prod_field: xr.DataArray, ref_field: xr.DataArray, test: str):
    if not SCIPY_OK:
        return np.nan
    vp, vr = flatten_pair(prod_field, ref_field)
    if vp.size < 10:
        return np.nan

    d = vp - vr
    if np.allclose(d, 0, atol=0, rtol=0):
        return 1.0

    if test == "wilcoxon":
        nz = d != 0
        d2 = d[nz]
        if d2.size < 10:
            return np.nan
        try:
            _, p = stats.wilcoxon(d2, zero_method="wilcox", alternative="two-sided", mode="auto")
            return float(p)
        except Exception:
            _, p = stats.ttest_rel(vp, vr, nan_policy="omit")
            return float(p) if np.isfinite(p) else np.nan
    else:
        _, p = stats.ttest_rel(vp, vr, nan_policy="omit")
        return float(p) if np.isfinite(p) else np.nan

def fdr_bh(pvals_1d: np.ndarray):
    p = np.array(pvals_1d, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    if pv.size == 0:
        return q

    order = np.argsort(pv)
    pv_sorted = pv[order]
    m = pv_sorted.size
    ranks = np.arange(1, m + 1)
    q_sorted = pv_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q_ok = np.empty_like(pv_sorted)
    q_ok[order] = q_sorted
    q[ok] = q_ok
    return q

def stars_from_q(q):
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


# =========================
# Plotting
# =========================
def plot_main_heatmap(score_df: pd.DataFrame, q_df: pd.DataFrame, out_png: str, title: str):
    import matplotlib.colors as mcolors
    try:
        import matplotlib.patheffects as pe
    except Exception:
        pe = None

    mat = score_df.values.astype(float)
    finite = mat[np.isfinite(mat)]

    # vmin/vmax strategy
    if SCORE_TYPE in ["bias", "corr", "nse"]:
        if SCORE_TYPE == "corr":
            vmin, vmax = -1.0, 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = "RdBu_r"
        elif SCORE_TYPE == "nse":
            # 以0为中心更直观：>0 好于参考均值基线，<0 更差
            if finite.size == 0:
                vlim = 1.0
            else:
                vlim = float(np.nanpercentile(np.abs(finite), HEATMAP_VMAX_PCTL))
                vlim = max(vlim, 1e-6)
            # NSE上界1，很多情况下下界可能<-1，用分位数截断更稳
            norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=min(1.0, vlim))
            cmap = "RdBu_r"
    else:
        # non-negative metrics
        if finite.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.nanpercentile(finite, HEATMAP_VMAX_PCTL))
            vmax = max(vmax, 1e-6)
        vmin = HEATMAP_VMIN if HEATMAP_VMIN is not None else 0.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = HEATMAP_CMAP

    fig = plt.figure(figsize=(1.0 + 0.75 * mat.shape[1], 1.2 + 0.45 * mat.shape[0]))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(score_df.shape[1]))
    ax.set_xticklabels(score_df.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(score_df.shape[0]))
    ax.set_yticklabels(score_df.index.tolist())
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(SCORE_TYPE)

    if HEATMAP_ANNOTATE:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                txt = f"{v:.3g}"
                if HEATMAP_ANNOTATE_STARS and (q_df is not None):
                    txt += stars_from_q(q_df.iloc[i, j])

                rgba = im.cmap(norm(v))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "black" if lum > 0.6 else "white"

                kw = dict(ha="center", va="center", fontsize=8, color=text_color)
                if HEATMAP_TEXT_BBOX:
                    kw["bbox"] = dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.25), ec="none")
                    kw["color"] = "white"

                t = ax.text(j, i, txt, **kw)
                if HEATMAP_TEXT_OUTLINE and (pe is not None) and (not HEATMAP_TEXT_BBOX):
                    edge = "black" if text_color == "white" else "white"
                    t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=edge)])

    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

def make_chain_df(score_df: pd.DataFrame):
    rows = []
    for idx in INDICES:
        fused_score = score_df.loc[FUSED_NAME, idx] if (FUSED_NAME in score_df.index) else np.nan
        for raw, bc in RAW_BC_PAIRS:
            raw_score = score_df.loc[raw, idx] if raw in score_df.index else np.nan
            bc_score  = score_df.loc[bc, idx] if bc in score_df.index else np.nan
            rows.append(dict(
                index=idx,
                pair=f"{raw}_to_{bc}",   # avoid unicode arrow in some terminals
                raw_name=raw,
                bc_name=bc,
                raw_score=raw_score,
                bc_score=bc_score,
                fused_score=fused_score
            ))
    return pd.DataFrame(rows)

def make_chain_improvement_matrices(chain_df: pd.DataFrame):
    pairs = chain_df["pair"].drop_duplicates().tolist()
    indices = chain_df["index"].drop_duplicates().tolist()
    imp1 = pd.DataFrame(index=pairs, columns=indices, dtype=float)  # RAW->BC
    imp2 = pd.DataFrame(index=pairs, columns=indices, dtype=float)  # BC->FUSED

    for pair in pairs:
        sub = chain_df[chain_df["pair"] == pair].set_index("index").reindex(indices)
        for idx in indices:
            raw = float(sub.loc[idx, "raw_score"]) if idx in sub.index else np.nan
            bc  = float(sub.loc[idx, "bc_score"]) if idx in sub.index else np.nan
            fu  = float(sub.loc[idx, "fused_score"]) if idx in sub.index else np.nan
            if better_is_lower(SCORE_TYPE):
                imp1.loc[pair, idx] = raw - bc if np.isfinite(raw) and np.isfinite(bc) else np.nan
                imp2.loc[pair, idx] = bc - fu if np.isfinite(bc) and np.isfinite(fu) else np.nan
            else:
                imp1.loc[pair, idx] = bc - raw if np.isfinite(raw) and np.isfinite(bc) else np.nan
                imp2.loc[pair, idx] = fu - bc if np.isfinite(bc) and np.isfinite(fu) else np.nan
    return imp1, imp2

def plot_improve_heatmap(improve_df: pd.DataFrame, out_png: str, title: str, vmax_pctl=95):
    import matplotlib.colors as mcolors
    try:
        import matplotlib.patheffects as pe
    except Exception:
        pe = None

    mat = improve_df.values.astype(float)
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        vlim = 1.0
    else:
        vlim = float(np.nanpercentile(np.abs(finite), vmax_pctl))
        vlim = max(vlim, 1e-6)

    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    cmap = "RdBu_r"  # positive (better) red, negative (worse) blue

    fig = plt.figure(figsize=(1.0 + 0.85 * mat.shape[1], 1.2 + 0.55 * mat.shape[0]))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(improve_df.shape[1]))
    ax.set_xticklabels(improve_df.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(improve_df.shape[0]))
    ax.set_yticklabels(improve_df.index.tolist())
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("improvement (+ better)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            rgba = im.cmap(norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "black" if lum > 0.6 else "white"
            t = ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=8, color=text_color)
            if HEATMAP_TEXT_OUTLINE and (pe is not None):
                edge = "black" if text_color == "white" else "white"
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=edge)])

    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

def plot_chain_bars(chain_df: pd.DataFrame, out_png: str, title: str, pair: str):
    sub = chain_df[chain_df["pair"] == pair].set_index("index").reindex(INDICES)
    indices = sub.index.tolist()

    raw = sub["raw_score"].values.astype(float)
    bc  = sub["bc_score"].values.astype(float)
    fu  = sub["fused_score"].values.astype(float)

    x = np.arange(len(indices))
    w = 0.28

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.bar(x - w, raw, width=w, label="RAW")
    ax.bar(x,     bc,  width=w, label="BC")
    ax.bar(x + w, fu,  width=w, label="FUSED")

    ax.set_xticks(x)
    ax.set_xticklabels(indices, rotation=30, ha="right")
    ax.set_ylabel(f"{SCORE_TYPE} vs {REF_PRODUCT}" + (" (normalized)" if (NORMALIZE_BY_REF_MEAN and allow_normalize(SCORE_TYPE)) else ""))
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    ensure_dir(OUT_DIR)
    fig_dir = os.path.join(OUT_DIR, "figs")
    tab_dir = os.path.join(OUT_DIR, "tables")
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    # region mask
    region_mask = None
    if USE_SHP_MASK:
        for idx in INDICES:
            fp0 = build_fp(REF_PRODUCT, idx)
            if os.path.exists(fp0):
                ds0 = xr.open_dataset(fp0)
                v0 = pick_data_var(ds0, index_name=idx)
                region_mask = build_region_mask(ds0[v0])
                ds0.close()
                break

    # REF climatology fields + REF means for normalization
    ref_fields = {}
    ref_means = {}

    for idx in INDICES:
        fp = build_fp(REF_PRODUCT, idx)
        if not os.path.exists(fp):
            print(f"[WARN] Missing REF file: {fp}")
            ref_fields[idx] = None
            ref_means[idx] = np.nan
            continue

        ds = xr.open_dataset(fp)
        v = pick_data_var(ds, index_name=idx)
        da = climatology_field(ds[v], region_mask=region_mask)
        ds.close()

        ref_fields[idx] = da
        vals = da.values.astype(float).ravel()
        vals = vals[np.isfinite(vals)]
        ref_means[idx] = float(np.nanmean(vals)) if vals.size else np.nan

    # score matrix + p matrix
    score_df = pd.DataFrame(index=PRODUCTS, columns=INDICES, dtype=float)
    p_df = pd.DataFrame(index=PRODUCTS, columns=INDICES, dtype=float)

    for prod in PRODUCTS:
        for idx in INDICES:
            ref_da = ref_fields.get(idx, None)
            if ref_da is None:
                score_df.loc[prod, idx] = np.nan
                p_df.loc[prod, idx] = np.nan
                continue

            fp = build_fp(prod, idx)
            if not os.path.exists(fp):
                print(f"[WARN] Missing: {fp}")
                score_df.loc[prod, idx] = np.nan
                p_df.loc[prod, idx] = np.nan
                continue

            ds = xr.open_dataset(fp)
            v = pick_data_var(ds, index_name=idx)
            prod_da = climatology_field(ds[v], region_mask=region_mask)
            ds.close()

            s = spatial_score(prod_da, ref_da, SCORE_TYPE)
            s = normalize_score(s, ref_means.get(idx, np.nan), SCORE_TYPE)
            score_df.loc[prod, idx] = s

            if SIGNIFICANCE_TESTS and (prod != REF_PRODUCT):
                p_df.loc[prod, idx] = paired_pvalue(prod_da, ref_da, test=PAIRED_TEST)
            else:
                p_df.loc[prod, idx] = np.nan

            print(f"[OK] {prod} | {idx} | score={score_df.loc[prod, idx]}")

    p_df.loc[REF_PRODUCT, :] = np.nan

    # FDR q-values
    q_df = p_df.copy()
    if SIGNIFICANCE_TESTS:
        flat_q = fdr_bh(p_df.values.ravel())
        q_df.values[:] = flat_q.reshape(p_df.shape)
    else:
        q_df[:] = np.nan

    # save tables
    score_df.to_csv(os.path.join(tab_dir, f"scores_{SCORE_TYPE}_{FREQ}.csv"))
    p_df.to_csv(os.path.join(tab_dir, f"pvalues_{PAIRED_TEST}_{FREQ}.csv"))
    q_df.to_csv(os.path.join(tab_dir, f"qvalues_fdr_{FREQ}.csv"))

    # main heatmap
    if MAKE_MAIN_HEATMAP:
        title = f"{len(PRODUCTS)}x{len(INDICES)} heatmap ({SCORE_TYPE} vs {REF_PRODUCT}) | {FREQ}"
        if NORMALIZE_BY_REF_MEAN and allow_normalize(SCORE_TYPE):
            title += " | normalized by |REF mean| (pctl clip)"
        plot_main_heatmap(
            score_df,
            q_df if SIGNIFICANCE_TESTS else None,
            os.path.join(fig_dir, f"heatmap_{SCORE_TYPE}_{FREQ}.png"),
            title
        )

    # chain analysis: improvement heatmaps + per-pair bars
    chain_df = make_chain_df(score_df)
    chain_df.to_csv(os.path.join(tab_dir, f"chain_raw_bc_fused_{SCORE_TYPE}_{FREQ}.csv"), index=False)

    if DO_CHAIN_IMPROVEMENT_HEATMAPS:
        imp1, imp2 = make_chain_improvement_matrices(chain_df)
        imp1.to_csv(os.path.join(tab_dir, f"improve_RAW_to_BC_{SCORE_TYPE}_{FREQ}.csv"))
        imp2.to_csv(os.path.join(tab_dir, f"improve_BC_to_FUSED_{SCORE_TYPE}_{FREQ}.csv"))

        plot_improve_heatmap(
            imp1,
            os.path.join(fig_dir, f"heat_improve_RAW_to_BC_{SCORE_TYPE}_{FREQ}.png"),
            f"Improvement RAW_to_BC ({SCORE_TYPE} vs {REF_PRODUCT}) | {FREQ}"
        )
        plot_improve_heatmap(
            imp2,
            os.path.join(fig_dir, f"heat_improve_BC_to_FUSED_{SCORE_TYPE}_{FREQ}.png"),
            f"Improvement BC_to_FUSED ({SCORE_TYPE} vs {REF_PRODUCT}) | {FREQ}"
        )

    if DO_CHAIN_BAR_PLOTS:
        for pair in chain_df["pair"].drop_duplicates().tolist():
            outp = os.path.join(fig_dir, f"bars_chain_{pair}_{SCORE_TYPE}_{FREQ}.png")
            plot_chain_bars(
                chain_df,
                outp,
                f"RAW vs BC vs FUSED | {pair} | {SCORE_TYPE} | {FREQ}",
                pair
            )

    print("\n[ALL DONE]")
    print("Tables:", tab_dir)
    print("Figs  :", fig_dir)


if __name__ == "__main__":
    main()
