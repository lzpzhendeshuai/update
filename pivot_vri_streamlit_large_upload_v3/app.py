# -*- coding: utf-8 -*-
"""
指针式喷灌机变量灌溉处方图生成工具
Streamlit 大图上传优化版 v3

核心改造：
1. 上传后保存为服务器临时文件，避免 MemoryFile 一次性读全图。
2. 使用 rasterio window 分块读取 GeoTIFF。
3. 第一遍抽样估算指数分位数，第二遍分块聚合处方矩阵。
4. 直接输出“角度扇区 × 喷头编号”矩阵，不生成完整像元点表。
5. Blue 波段可选，Thermal 波段可选。
"""

from __future__ import annotations

import io
import math
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import rasterio
    from rasterio.windows import Window
except Exception:  # pragma: no cover
    rasterio = None
    Window = None


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class MachineConfig:
    span_length_m: float
    span_count: int
    sprinkler_spacing_m: float
    first_sprinkler_radius_m: float
    sector_deg: float
    spray_radius_m: float
    radial_width_m: float
    rotation_time_100_hr: float
    speed_levels_percent: List[float]
    pwm_period_s: float
    duty_levels_percent: List[float]
    min_effective_duty_percent: float
    target_standard_depth_100_mm: float
    flow_mode: str
    fixed_flow_lpm: float
    weighting_mode: str
    center_x: float
    center_y: float


@dataclass
class RemoteSensingConfig:
    reflectance_scale: float
    coordinate_mode: str
    pixel_size_m: float
    sample_step: int
    stats_sample_step: int
    block_size: int
    ndvi_threshold: float
    max_depth_mm: float
    min_depth_mm: float
    p_low: float
    p_high: float
    w_ndvi: float
    w_ndre: float
    w_gndvi: float
    w_evi: float
    w_thermal: float
    thermal_is_celsius: bool
    use_manual_cwsi: bool
    wet_temp_c: float
    dry_temp_c: float
    stretch_stress: bool
    non_veg_to_zero: bool


# =============================================================================
# 通用工具
# =============================================================================

def parse_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for item in text.replace("，", ",").split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return sorted(vals)


def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float32)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    out[mask] = num[mask] / den[mask]
    return out


def clean_optional_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = str(text).strip().strip('"').strip("'")
    return t if t else None


def iter_windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row_off in range(0, height, block_size):
        for col_off in range(0, width, block_size):
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=min(block_size, width - col_off),
                height=min(block_size, height - row_off),
            )


def save_uploaded_to_temp(uploaded_file, temp_dir: str, label: str) -> str:
    """把 Streamlit 上传文件保存到服务器临时目录，返回本地路径。"""
    safe_name = Path(uploaded_file.name).name
    if not safe_name.lower().endswith((".tif", ".tiff")):
        safe_name += ".tif"
    out_path = os.path.join(temp_dir, f"{label}_{safe_name}")
    uploaded_file.seek(0)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return out_path


def read_window_band(src, window: Window, scale: float = 1.0, sample_step: int = 1, convert_to_float: bool = True) -> np.ndarray:
    arr = src.read(1, window=window)
    if convert_to_float:
        arr = arr.astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        if scale and abs(scale - 1.0) > 1e-12:
            arr = arr / float(scale)
    if sample_step > 1:
        arr = arr[::sample_step, ::sample_step]
    return arr.astype(np.float32, copy=False)


def window_xy_arrays(src, window: Window, arr_shape: Tuple[int, int], sample_step: int, rs_cfg: RemoteSensingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """计算窗口抽样像元中心的 X/Y 坐标。"""
    h, w = arr_shape
    rows = int(window.row_off) + np.arange(0, int(window.height), sample_step)[:h]
    cols = int(window.col_off) + np.arange(0, int(window.width), sample_step)[:w]
    cc, rr = np.meshgrid(cols, rows)

    if rs_cfg.coordinate_mode == "使用 GeoTIFF 地理坐标，单位应为 m":
        transform = src.transform
        x = transform.c + transform.a * (cc + 0.5) + transform.b * (rr + 0.5)
        y = transform.f + transform.d * (cc + 0.5) + transform.e * (rr + 0.5)
    else:
        px = float(rs_cfg.pixel_size_m)
        x = (cc + 0.5) * px
        y = (src.height - rr - 0.5) * px
    return x.astype(np.float32), y.astype(np.float32)


def robust_norm_by_percentile(x: np.ndarray, p_low: float, p_high: float, inverse: bool = False) -> np.ndarray:
    out = (x - p_low) / max(p_high - p_low, 1e-6)
    out = np.clip(out, 0.0, 1.0)
    if inverse:
        out = 1.0 - out
    return out.astype(np.float32)


def validate_raster_shapes(paths: Dict[str, Optional[str]]) -> dict:
    """检查各波段尺寸、坐标和分辨率。返回参考影像元数据。"""
    required = ["red", "green", "rededge", "nir"]
    for key in required:
        if not paths.get(key):
            raise ValueError(f"缺少必选波段：{key}")

    meta = {}
    ref_shape = None
    ref_transform = None
    ref_crs = None
    ref_res = None

    for key, path in paths.items():
        if not path:
            continue
        with rasterio.open(path) as src:
            shape = (src.height, src.width)
            if ref_shape is None:
                ref_shape = shape
                ref_transform = src.transform
                ref_crs = src.crs
                ref_res = src.res
                meta = {
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs) if src.crs else "None",
                    "transform": str(src.transform),
                    "res": src.res,
                    "bounds": tuple(src.bounds),
                }
            else:
                if shape != ref_shape:
                    raise ValueError(f"{key} 影像尺寸 {shape} 与参考影像 {ref_shape} 不一致，请先重采样对齐。")
                # 只警告，不直接报错。某些软件导出的 transform 会有微小浮点差。
                if src.res != ref_res:
                    st.warning(f"{key} 像元大小 {src.res} 与参考影像 {ref_res} 不一致，建议先统一重采样。")
                if src.crs != ref_crs:
                    st.warning(f"{key} 坐标系与参考影像不一致，建议先统一投影。")
                if src.transform != ref_transform:
                    st.warning(f"{key} 仿射变换与参考影像不完全一致，若波段未严格配准，指数计算会有偏差。")
    return meta


# =============================================================================
# 喷头、扇区和控制表
# =============================================================================

def generate_sprinklers(cfg: MachineConfig) -> pd.DataFrame:
    total_radius = cfg.span_length_m * cfg.span_count
    radii = np.arange(cfg.first_sprinkler_radius_m, total_radius + 0.001, cfg.sprinkler_spacing_m, dtype=float)
    omega_100 = 2.0 * math.pi / (cfg.rotation_time_100_hr * 60.0)

    if cfg.flow_mode == "自动按半径匹配流量，使100%速度全开时近似等深灌溉":
        flows = cfg.target_standard_depth_100_mm * omega_100 * radii * cfg.radial_width_m
        flows = np.maximum(flows, 0.01)
    else:
        flows = np.full_like(radii, cfg.fixed_flow_lpm, dtype=float)

    return pd.DataFrame({
        "sprinkler_id": [f"P{i + 1:03d}" for i in range(len(radii))],
        "radius_m": radii,
        "flow_lpm": flows,
        "spray_radius_m": cfg.spray_radius_m,
        "radial_width_m": cfg.radial_width_m,
    })


def sector_table(cfg: MachineConfig) -> pd.DataFrame:
    starts = np.arange(0.0, 360.0, cfg.sector_deg)
    ends = np.minimum(starts + cfg.sector_deg, 360.0)
    mids = (starts + ends) / 2.0
    return pd.DataFrame({
        "sector_id": np.arange(1, len(starts) + 1),
        "angle_start_deg": starts,
        "angle_end_deg": ends,
        "angle_mid_deg": mids,
    })


def select_sector_speed_and_duty(polar_df: pd.DataFrame, cfg: MachineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    omega_100 = 2.0 * math.pi / (cfg.rotation_time_100_hr * 60.0)
    speed_levels = sorted(cfg.speed_levels_percent)
    duty_levels = np.array(sorted(cfg.duty_levels_percent), dtype=float)

    out_records = []
    speed_records = []

    for sector_id, g in polar_df.groupby("sector_id", sort=True):
        candidate_stats = []
        for spd in speed_levels:
            omega = omega_100 * spd / 100.0
            duty = (
                g["target_mm"].to_numpy(dtype=float)
                * omega
                * g["radius_m"].to_numpy(dtype=float)
                * g["radial_width_m"].to_numpy(dtype=float)
                / np.maximum(g["flow_lpm"].to_numpy(dtype=float), 1e-6)
            )
            p95 = float(np.nanpercentile(duty, 95)) if len(duty) else 0.0
            pmax = float(np.nanmax(duty)) if len(duty) else 0.0
            candidate_stats.append((spd, p95, pmax))

        feasible = [item for item in candidate_stats if item[1] <= 1.0]
        selected_speed = max([item[0] for item in feasible]) if feasible else min(speed_levels)

        sec_meta = g.iloc[0]
        speed_records.append({
            "sector_id": int(sector_id),
            "angle_start_deg": float(sec_meta["angle_start_deg"]),
            "angle_end_deg": float(sec_meta["angle_end_deg"]),
            "angle_mid_deg": float(sec_meta["angle_mid_deg"]),
            "target_mean_mm": float(g["target_mm"].mean()),
            "target_p95_mm": float(g["target_mm"].quantile(0.95)),
            "speed_percent": float(selected_speed),
        })

        omega = omega_100 * selected_speed / 100.0
        for _, row in g.iterrows():
            raw_duty = row["target_mm"] * omega * row["radius_m"] * row["radial_width_m"] / max(row["flow_lpm"], 1e-6)
            raw_duty_percent = float(np.clip(raw_duty * 100.0, 0.0, 100.0))
            if raw_duty_percent < cfg.min_effective_duty_percent:
                quantized = 0.0
            else:
                quantized = float(duty_levels[np.argmin(np.abs(duty_levels - raw_duty_percent))])
            on_seconds = cfg.pwm_period_s * quantized / 100.0
            off_seconds = cfg.pwm_period_s - on_seconds
            out_records.append({
                "sector_id": int(sector_id),
                "angle_start_deg": float(row["angle_start_deg"]),
                "angle_end_deg": float(row["angle_end_deg"]),
                "angle_mid_deg": float(row["angle_mid_deg"]),
                "speed_percent": float(selected_speed),
                "sprinkler_id": row["sprinkler_id"],
                "radius_m": float(row["radius_m"]),
                "flow_lpm": float(row["flow_lpm"]),
                "radial_width_m": float(row["radial_width_m"]),
                "target_mm": float(row["target_mm"]),
                "raw_duty_percent": raw_duty_percent,
                "duty_percent": quantized,
                "on_seconds": float(on_seconds),
                "off_seconds": float(off_seconds),
            })

    return pd.DataFrame(out_records), pd.DataFrame(speed_records)


def make_wide_matrix(control_long: pd.DataFrame) -> pd.DataFrame:
    base = control_long[["sector_id", "angle_start_deg", "angle_end_deg", "speed_percent"]].drop_duplicates()
    pivot = control_long.pivot_table(index="sector_id", columns="sprinkler_id", values="duty_percent", aggfunc="mean").reset_index()
    out = base.merge(pivot, on="sector_id", how="left")
    return out.sort_values("sector_id")


# =============================================================================
# 大影像两遍分块处理
# =============================================================================

def compute_indices_from_bands(red, green, rededge, nir, blue=None):
    ndvi = safe_div(nir - red, nir + red)
    ndre = safe_div(nir - rededge, nir + rededge)
    gndvi = safe_div(nir - green, nir + green)
    if blue is not None:
        evi = safe_div(2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0)
        evi_name = "EVI"
    else:
        evi = safe_div(2.5 * (nir - red), nir + 2.4 * red + 1.0)
        evi_name = "EVI2"
    return ndvi, ndre, gndvi, evi, evi_name


def collect_index_percentiles(paths: Dict[str, Optional[str]], rs_cfg: RemoteSensingConfig, progress=None) -> pd.DataFrame:
    """第一遍：抽样估算指数和温度分位数。"""
    samples: Dict[str, List[np.ndarray]] = {"NDVI": [], "NDRE": [], "GNDVI": [], "EVI": [], "EVI2": [], "Tc_C": []}
    max_samples_per_index = 600_000

    with rasterio.open(paths["red"]) as red_src, \
         rasterio.open(paths["green"]) as green_src, \
         rasterio.open(paths["rededge"]) as rededge_src, \
         rasterio.open(paths["nir"]) as nir_src:

        blue_src = rasterio.open(paths["blue"]) if paths.get("blue") else None
        thermal_src = rasterio.open(paths["thermal"]) if paths.get("thermal") else None
        windows = list(iter_windows(red_src.width, red_src.height, rs_cfg.block_size))

        try:
            for idx, window in enumerate(windows, start=1):
                red = read_window_band(red_src, window, rs_cfg.reflectance_scale, rs_cfg.stats_sample_step)
                green = read_window_band(green_src, window, rs_cfg.reflectance_scale, rs_cfg.stats_sample_step)
                rededge = read_window_band(rededge_src, window, rs_cfg.reflectance_scale, rs_cfg.stats_sample_step)
                nir = read_window_band(nir_src, window, rs_cfg.reflectance_scale, rs_cfg.stats_sample_step)
                blue = read_window_band(blue_src, window, rs_cfg.reflectance_scale, rs_cfg.stats_sample_step) if blue_src is not None else None

                ndvi, ndre, gndvi, evi, evi_name = compute_indices_from_bands(red, green, rededge, nir, blue)
                veg = np.isfinite(ndvi) & (ndvi >= rs_cfg.ndvi_threshold)

                if np.any(veg):
                    for name, arr in [("NDVI", ndvi), ("NDRE", ndre), ("GNDVI", gndvi), (evi_name, evi)]:
                        vals = arr[veg & np.isfinite(arr)]
                        if vals.size:
                            # 每块最多抽取一部分，避免样本堆得过多
                            if vals.size > 10000:
                                pick = np.linspace(0, vals.size - 1, 10000).astype(int)
                                vals = vals[pick]
                            samples[name].append(vals.astype(np.float32, copy=False))

                    if thermal_src is not None:
                        tc = read_window_band(thermal_src, window, 1.0, rs_cfg.stats_sample_step)
                        if not rs_cfg.thermal_is_celsius:
                            tc = tc - 273.15
                        vals = tc[veg & np.isfinite(tc)]
                        if vals.size:
                            if vals.size > 10000:
                                pick = np.linspace(0, vals.size - 1, 10000).astype(int)
                                vals = vals[pick]
                            samples["Tc_C"].append(vals.astype(np.float32, copy=False))

                if progress is not None:
                    progress.progress(idx / len(windows) * 0.35, text=f"第一遍抽样估算分位数：{idx}/{len(windows)} 块")

                # 总样本上限
                for key, arrs in samples.items():
                    total = sum(a.size for a in arrs)
                    if total > max_samples_per_index:
                        samples[key] = [np.concatenate(arrs)[:: max(1, total // max_samples_per_index)]]

        finally:
            if blue_src is not None:
                blue_src.close()
            if thermal_src is not None:
                thermal_src.close()

    records = []
    for name, arrs in samples.items():
        if not arrs:
            continue
        vals = np.concatenate(arrs)
        vals = vals[np.isfinite(vals)]
        if vals.size < 10:
            continue
        records.append({
            "index": name,
            "p_low": float(np.nanpercentile(vals, rs_cfg.p_low)),
            "p_high": float(np.nanpercentile(vals, rs_cfg.p_high)),
            "p05": float(np.nanpercentile(vals, 5)),
            "p50": float(np.nanpercentile(vals, 50)),
            "p95": float(np.nanpercentile(vals, 95)),
            "mean": float(np.nanmean(vals)),
            "sample_n": int(vals.size),
        })
    return pd.DataFrame(records)


def stats_lookup(stats_df: pd.DataFrame, index_name: str) -> Optional[Tuple[float, float]]:
    sub = stats_df[stats_df["index"] == index_name]
    if sub.empty:
        return None
    return float(sub.iloc[0]["p_low"]), float(sub.iloc[0]["p_high"])


def process_big_raster_to_polar(
    paths: Dict[str, Optional[str]],
    cfg: MachineConfig,
    rs_cfg: RemoteSensingConfig,
    sprinklers: pd.DataFrame,
    sectors: pd.DataFrame,
    stats_df: pd.DataFrame,
    progress=None,
) -> pd.DataFrame:
    """第二遍：分块计算综合胁迫和灌水量，并直接聚合到 sector × sprinkler。"""
    n_sectors = len(sectors)
    n_sprinklers = len(sprinklers)
    sum_depth_weight = np.zeros((n_sectors, n_sprinklers), dtype=np.float64)
    sum_weight = np.zeros((n_sectors, n_sprinklers), dtype=np.float64)

    weights_cfg = {
        "NDVI": rs_cfg.w_ndvi,
        "NDRE": rs_cfg.w_ndre,
        "GNDVI": rs_cfg.w_gndvi,
        "EVI": rs_cfg.w_evi,
        "EVI2": rs_cfg.w_evi,
        "CWSI": rs_cfg.w_thermal,
    }

    stat_ranges = {name: stats_lookup(stats_df, name) for name in ["NDVI", "NDRE", "GNDVI", "EVI", "EVI2", "Tc_C"]}
    evi_name = "EVI" if paths.get("blue") else "EVI2"

    if paths.get("thermal") and rs_cfg.use_manual_cwsi:
        twet, tdry = rs_cfg.wet_temp_c, rs_cfg.dry_temp_c
    elif paths.get("thermal") and stat_ranges.get("Tc_C") is not None:
        # 用统计表中的 5% 和 95% 作为 CWSI 湿/干端元，更稳一些
        tc_row = stats_df[stats_df["index"] == "Tc_C"].iloc[0]
        twet, tdry = float(tc_row["p05"]), float(tc_row["p95"])
    else:
        twet, tdry = np.nan, np.nan

    with rasterio.open(paths["red"]) as red_src, \
         rasterio.open(paths["green"]) as green_src, \
         rasterio.open(paths["rededge"]) as rededge_src, \
         rasterio.open(paths["nir"]) as nir_src:

        blue_src = rasterio.open(paths["blue"]) if paths.get("blue") else None
        thermal_src = rasterio.open(paths["thermal"]) if paths.get("thermal") else None
        windows = list(iter_windows(red_src.width, red_src.height, rs_cfg.block_size))
        rmax = cfg.span_length_m * cfg.span_count

        radii = sprinklers["radius_m"].to_numpy(dtype=np.float32)
        spray_radii = sprinklers["spray_radius_m"].to_numpy(dtype=np.float32)
        sector_meta = sectors.set_index("sector_id")

        try:
            for idx, window in enumerate(windows, start=1):
                red = read_window_band(red_src, window, rs_cfg.reflectance_scale, rs_cfg.sample_step)
                green = read_window_band(green_src, window, rs_cfg.reflectance_scale, rs_cfg.sample_step)
                rededge = read_window_band(rededge_src, window, rs_cfg.reflectance_scale, rs_cfg.sample_step)
                nir = read_window_band(nir_src, window, rs_cfg.reflectance_scale, rs_cfg.sample_step)
                blue = read_window_band(blue_src, window, rs_cfg.reflectance_scale, rs_cfg.sample_step) if blue_src is not None else None

                ndvi, ndre, gndvi, evi, evi_name_block = compute_indices_from_bands(red, green, rededge, nir, blue)
                veg = np.isfinite(ndvi) & (ndvi >= rs_cfg.ndvi_threshold)

                stress_sum = np.zeros(ndvi.shape, dtype=np.float32)
                wsum = np.zeros(ndvi.shape, dtype=np.float32)

                for name, arr, inverse in [
                    ("NDVI", ndvi, True),
                    ("NDRE", ndre, True),
                    ("GNDVI", gndvi, True),
                    (evi_name, evi, True),
                ]:
                    stat = stat_ranges.get(name)
                    w = float(weights_cfg.get(name, 0.0))
                    if stat is None or w <= 0:
                        continue
                    layer = robust_norm_by_percentile(arr, stat[0], stat[1], inverse=inverse)
                    valid = veg & np.isfinite(layer)
                    stress_sum[valid] += w * layer[valid]
                    wsum[valid] += w

                if thermal_src is not None:
                    tc = read_window_band(thermal_src, window, 1.0, rs_cfg.sample_step)
                    if not rs_cfg.thermal_is_celsius:
                        tc = tc - 273.15
                    if np.isfinite(twet) and np.isfinite(tdry) and abs(tdry - twet) > 1e-6:
                        cwsi = np.clip((tc - twet) / (tdry - twet), 0.0, 1.0).astype(np.float32)
                        valid = veg & np.isfinite(cwsi)
                        w = float(rs_cfg.w_thermal)
                        stress_sum[valid] += w * cwsi[valid]
                        wsum[valid] += w

                score = np.full(ndvi.shape, np.nan, dtype=np.float32)
                ok = veg & (wsum > 0)
                score[ok] = stress_sum[ok] / np.maximum(wsum[ok], 1e-6)
                score = np.clip(score, 0.0, 1.0)

                # 可选：分块内轻微拉伸，避免极端值影响。正式版建议用全局分位数建模。
                if rs_cfg.stretch_stress:
                    vals = score[ok & np.isfinite(score)]
                    if vals.size > 20:
                        lo = np.nanpercentile(vals, 2)
                        hi = np.nanpercentile(vals, 98)
                        if abs(hi - lo) > 1e-6:
                            score = np.clip((score - lo) / (hi - lo), 0.0, 1.0)

                water_mm = rs_cfg.min_depth_mm + score * (rs_cfg.max_depth_mm - rs_cfg.min_depth_mm)
                if rs_cfg.non_veg_to_zero:
                    water_mm[~veg] = 0.0
                water_mm[~np.isfinite(water_mm)] = 0.0

                x, y = window_xy_arrays(red_src, window, water_mm.shape, rs_cfg.sample_step, rs_cfg)
                dx = x - cfg.center_x
                dy = y - cfg.center_y
                r = np.sqrt(dx * dx + dy * dy).astype(np.float32)
                theta = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
                sector_idx = np.floor(theta / cfg.sector_deg).astype(np.int32)
                sector_idx = np.clip(sector_idx, 0, n_sectors - 1)

                valid_base = np.isfinite(water_mm) & np.isfinite(r) & (r <= rmax)
                if not np.any(valid_base):
                    if progress is not None:
                        progress.progress(0.35 + idx / len(windows) * 0.60, text=f"第二遍分块生成处方矩阵：{idx}/{len(windows)} 块")
                    continue

                # 对每个喷头做径向影响带聚合。该方法比逐圆相交快，适合 Web 端部署。
                for j in range(n_sprinklers):
                    ri = radii[j]
                    sr = spray_radii[j]
                    radial_dist = np.abs(r - ri)
                    mask = valid_base & (radial_dist <= sr)
                    if not np.any(mask):
                        continue
                    if cfg.weighting_mode == "均匀权重：圆内像元等权":
                        ww = np.ones(np.count_nonzero(mask), dtype=np.float32)
                    else:
                        sigma = max(float(sr) / 2.0, 1e-6)
                        ww = np.exp(-0.5 * (radial_dist[mask] / sigma) ** 2).astype(np.float32)

                    sec = sector_idx[mask]
                    depth = water_mm[mask].astype(np.float32, copy=False)
                    np.add.at(sum_depth_weight[:, j], sec, depth * ww)
                    np.add.at(sum_weight[:, j], sec, ww)

                if progress is not None:
                    progress.progress(0.35 + idx / len(windows) * 0.60, text=f"第二遍分块生成处方矩阵：{idx}/{len(windows)} 块")

        finally:
            if blue_src is not None:
                blue_src.close()
            if thermal_src is not None:
                thermal_src.close()

    target_mm = np.divide(sum_depth_weight, sum_weight, out=np.zeros_like(sum_depth_weight), where=sum_weight > 0)

    records = []
    for s_idx, sec in sectors.iterrows():
        for j, sp in sprinklers.iterrows():
            records.append({
                "sector_id": int(sec["sector_id"]),
                "angle_start_deg": float(sec["angle_start_deg"]),
                "angle_end_deg": float(sec["angle_end_deg"]),
                "angle_mid_deg": float(sec["angle_mid_deg"]),
                "sprinkler_id": sp["sprinkler_id"],
                "radius_m": float(sp["radius_m"]),
                "flow_lpm": float(sp["flow_lpm"]),
                "radial_width_m": float(sp["radial_width_m"]),
                "target_mm": float(target_mm[s_idx, j]),
            })

    if progress is not None:
        progress.progress(0.97, text="正在整理处方矩阵和控制表...")
    return pd.DataFrame(records)


# =============================================================================
# 绘图和下载
# =============================================================================

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: Dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return buffer.getvalue()


def plot_duty_heatmap(control_long: pd.DataFrame):
    matrix = control_long.pivot_table(index="sprinkler_id", columns="sector_id", values="duty_percent", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=130)
    im = ax.imshow(matrix.values, aspect="auto", origin="lower")
    ax.set_xlabel("Sector ID")
    ax.set_ylabel("Sprinkler ID")
    ax.set_title("Duty-cycle prescription matrix")
    x_ticks = np.linspace(0, matrix.shape[1] - 1, min(10, matrix.shape[1])).astype(int)
    y_ticks = np.linspace(0, matrix.shape[0] - 1, min(12, matrix.shape[0])).astype(int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(matrix.columns[x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(matrix.index[y_ticks])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Duty cycle / %")
    fig.tight_layout()
    return fig


def plot_speed(speed_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=130)
    ax.plot(speed_df["angle_mid_deg"], speed_df["speed_percent"], marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("Angle / degree")
    ax.set_ylabel("Travel speed / %")
    ax.set_title("Sector-level travel speed prescription")
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_polar_target(polar_df: pd.DataFrame):
    matrix = polar_df.pivot_table(index="sprinkler_id", columns="sector_id", values="target_mm", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=130)
    im = ax.imshow(matrix.values, aspect="auto", origin="lower")
    ax.set_xlabel("Sector ID")
    ax.set_ylabel("Sprinkler ID")
    ax.set_title("Polar target irrigation depth matrix")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Target depth / mm")
    fig.tight_layout()
    return fig


def uploaded_size_mb(f) -> float:
    if f is None:
        return 0.0
    try:
        return f.size / 1024 / 1024
    except Exception:
        return 0.0


# =============================================================================
# Streamlit 页面
# =============================================================================

st.set_page_config(page_title="指针式喷灌机变量灌溉处方图 大图版", layout="wide")
st.title("指针式喷灌机变量灌溉处方图生成工具：Streamlit 大图上传优化版")
st.caption("上传无人机多光谱/热红外 GeoTIFF 后，采用临时文件保存 + 分块读取 + 直接极坐标聚合，降低大影像处理压力。")

if rasterio is None:
    st.error("当前环境缺少 rasterio。请先运行：pip install rasterio")
    st.stop()

st.warning(
    "说明：本版本能优化上传后的读取和计算速度，但浏览器上传大文件仍受网络、Streamlit 会话和云端资源限制。"
    "几 GB 级正射影像正式产品建议走阿里云 OSS 分片上传 + ECS 后台计算。"
)

with st.sidebar:
    st.header("1. 机器参数")
    span_length_m = st.number_input("每跨长度 m", min_value=1.0, value=55.0, step=1.0)
    span_count = st.number_input("跨数", min_value=1, value=1, step=1)
    sprinkler_spacing_m = st.number_input("喷头间距 m", min_value=0.5, value=3.0, step=0.5)
    first_sprinkler_radius_m = st.number_input("第一个喷头距中心点 m", min_value=0.1, value=3.0, step=0.5)
    spray_radius_m = st.number_input("单喷头圆形喷洒半径 m", min_value=0.5, value=4.5, step=0.5)
    radial_width_m = st.number_input("单喷头代表径向宽度 m", min_value=0.5, value=3.0, step=0.5)

    st.header("2. 控制参数")
    sector_deg = st.selectbox("角度分区 deg", [0.5, 1.0, 2.0, 5.0], index=1)
    rotation_time_100_hr = st.number_input("100%速度完整转一圈时间 h", min_value=0.1, value=24.0, step=0.5)
    speed_levels_text = st.text_input("速度档位 %，逗号分隔", value="40,60,80,100")
    pwm_period_s = st.number_input("PWM控制周期 s", min_value=5.0, value=30.0, step=5.0)
    duty_levels_text = st.text_input("占空比档位 %，逗号分隔", value="0,25,50,75,100")
    min_effective_duty_percent = st.number_input("低于该占空比则关阀 %", min_value=0.0, value=5.0, step=1.0)

    st.header("3. 喷头流量")
    flow_mode = st.radio(
        "喷头流量设置",
        ["自动按半径匹配流量，使100%速度全开时近似等深灌溉", "所有喷头使用同一固定流量"],
        index=0,
    )
    target_standard_depth_100_mm = st.number_input("自动流量时：100%速度全开标准灌水深 mm", min_value=0.1, value=12.0, step=1.0)
    fixed_flow_lpm = st.number_input("固定流量 L/min", min_value=0.01, value=2.0, step=0.1)

    st.header("4. 坐标和喷洒核")
    weighting_mode = st.radio("喷洒核权重", ["均匀权重：圆内像元等权", "高斯权重：中心权重大、边缘权重小"], index=1)
    coordinate_mode = st.radio("坐标模式", ["使用本地像元尺寸构建米坐标", "使用 GeoTIFF 地理坐标，单位应为 m"], index=0)
    pixel_size_m = st.number_input("本地坐标模式：像元大小 m", min_value=0.01, value=0.50, step=0.05)
    center_x = st.number_input("支轴中心 X 坐标 m", value=0.0, step=1.0)
    center_y = st.number_input("支轴中心 Y 坐标 m", value=0.0, step=1.0)

    st.header("5. 大图处理参数")
    block_size = st.selectbox("分块大小", [512, 1024, 2048], index=1)
    sample_step = st.number_input("正式计算抽样步长，越大越快", min_value=1, max_value=100, value=5, step=1)
    stats_sample_step = st.number_input("第一遍分位数抽样步长", min_value=1, max_value=200, value=20, step=1)
    reflectance_scale = st.number_input("反射率缩放系数，0-1填1，0-10000填10000", min_value=1.0, value=10000.0, step=1000.0)

    st.header("6. 遥感反演参数")
    ndvi_threshold = st.number_input("植被掩膜 NDVI 阈值", min_value=-1.0, max_value=1.0, value=0.25, step=0.05)
    min_depth_mm = st.number_input("最低灌水量 mm", min_value=0.0, value=0.0, step=1.0)
    max_depth_mm = st.number_input("最高灌水量 mm", min_value=0.1, value=30.0, step=1.0)
    p_low = st.number_input("指数归一化低百分位", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    p_high = st.number_input("指数归一化高百分位", min_value=50.0, max_value=100.0, value=95.0, step=1.0)
    stretch_stress = st.checkbox("分块内对综合胁迫轻微拉伸", value=False)
    non_veg_to_zero = st.checkbox("非植被区域灌水量设为 0", value=True)

    st.markdown("**指数权重**")
    w_ndvi = st.number_input("NDVI 胁迫权重", min_value=0.0, value=0.30, step=0.05)
    w_ndre = st.number_input("NDRE 胁迫权重", min_value=0.0, value=0.30, step=0.05)
    w_gndvi = st.number_input("GNDVI 胁迫权重", min_value=0.0, value=0.20, step=0.05)
    w_evi = st.number_input("EVI/EVI2 胁迫权重", min_value=0.0, value=0.10, step=0.05)
    w_thermal = st.number_input("热红外 CWSI 权重，有热红外时生效", min_value=0.0, value=0.40, step=0.05)

    st.markdown("**热红外参数**")
    thermal_is_celsius = st.checkbox("热红外输入已经是摄氏度 ℃", value=True)
    use_manual_cwsi = st.checkbox("手动设置湿/干参考温度", value=False)
    wet_temp_c = st.number_input("湿参考温度 ℃", value=22.0, step=0.5)
    dry_temp_c = st.number_input("干参考温度 ℃", value=38.0, step=0.5)

try:
    cfg = MachineConfig(
        span_length_m=float(span_length_m),
        span_count=int(span_count),
        sprinkler_spacing_m=float(sprinkler_spacing_m),
        first_sprinkler_radius_m=float(first_sprinkler_radius_m),
        sector_deg=float(sector_deg),
        spray_radius_m=float(spray_radius_m),
        radial_width_m=float(radial_width_m),
        rotation_time_100_hr=float(rotation_time_100_hr),
        speed_levels_percent=parse_float_list(speed_levels_text),
        pwm_period_s=float(pwm_period_s),
        duty_levels_percent=parse_float_list(duty_levels_text),
        min_effective_duty_percent=float(min_effective_duty_percent),
        target_standard_depth_100_mm=float(target_standard_depth_100_mm),
        flow_mode=flow_mode,
        fixed_flow_lpm=float(fixed_flow_lpm),
        weighting_mode=weighting_mode,
        center_x=float(center_x),
        center_y=float(center_y),
    )
    rs_cfg = RemoteSensingConfig(
        reflectance_scale=float(reflectance_scale),
        coordinate_mode=coordinate_mode,
        pixel_size_m=float(pixel_size_m),
        sample_step=int(sample_step),
        stats_sample_step=int(stats_sample_step),
        block_size=int(block_size),
        ndvi_threshold=float(ndvi_threshold),
        max_depth_mm=float(max_depth_mm),
        min_depth_mm=float(min_depth_mm),
        p_low=float(p_low),
        p_high=float(p_high),
        w_ndvi=float(w_ndvi),
        w_ndre=float(w_ndre),
        w_gndvi=float(w_gndvi),
        w_evi=float(w_evi),
        w_thermal=float(w_thermal),
        thermal_is_celsius=bool(thermal_is_celsius),
        use_manual_cwsi=bool(use_manual_cwsi),
        wet_temp_c=float(wet_temp_c),
        dry_temp_c=float(dry_temp_c),
        stretch_stress=bool(stretch_stress),
        non_veg_to_zero=bool(non_veg_to_zero),
    )
except Exception as e:
    st.error(f"参数解析失败：{e}")
    st.stop()

if not cfg.speed_levels_percent or not cfg.duty_levels_percent:
    st.error("速度档位和占空比档位不能为空。")
    st.stop()

st.subheader("1. 上传多光谱/热红外 GeoTIFF")
st.write("Red、Green、RedEdge、NIR 必选；Blue 可选；Thermal 可选。建议先在 GIS 软件中裁剪到指针机覆盖范围，并降采样到 0.3–1.0 m。")

with st.form("upload_and_run_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        red_file = st.file_uploader("Red 红波段 GeoTIFF", type=["tif", "tiff"], key="red")
        green_file = st.file_uploader("Green 绿波段 GeoTIFF", type=["tif", "tiff"], key="green")
    with col2:
        rededge_file = st.file_uploader("RedEdge 红边波段 GeoTIFF", type=["tif", "tiff"], key="rededge")
        nir_file = st.file_uploader("NIR 近红外波段 GeoTIFF", type=["tif", "tiff"], key="nir")
    with col3:
        blue_file = st.file_uploader("Blue 蓝波段 GeoTIFF，可选", type=["tif", "tiff"], key="blue")
        thermal_file = st.file_uploader("Thermal 热红外 GeoTIFF，可选", type=["tif", "tiff"], key="thermal")

    submitted = st.form_submit_button("开始生成大图处方图", type="primary")

uploaded_files = [red_file, green_file, rededge_file, nir_file, blue_file, thermal_file]
size_total = sum(uploaded_size_mb(f) for f in uploaded_files)
if size_total > 0:
    st.info(f"当前已选择文件总大小约 {size_total:.1f} MB。上传速度取决于网络和 Streamlit 服务器资源。")

if submitted:
    missing = []
    for name, f in [("Red", red_file), ("Green", green_file), ("RedEdge", rededge_file), ("NIR", nir_file)]:
        if f is None:
            missing.append(name)
    if missing:
        st.error("缺少必选波段：" + "、".join(missing))
        st.stop()

    t0 = time.time()
    temp_dir = tempfile.mkdtemp(prefix="pivot_vri_upload_")
    progress = st.progress(0.0, text="正在保存上传文件到服务器临时目录...")

    try:
        paths = {
            "red": save_uploaded_to_temp(red_file, temp_dir, "red"),
            "green": save_uploaded_to_temp(green_file, temp_dir, "green"),
            "rededge": save_uploaded_to_temp(rededge_file, temp_dir, "rededge"),
            "nir": save_uploaded_to_temp(nir_file, temp_dir, "nir"),
            "blue": save_uploaded_to_temp(blue_file, temp_dir, "blue") if blue_file is not None else None,
            "thermal": save_uploaded_to_temp(thermal_file, temp_dir, "thermal") if thermal_file is not None else None,
        }
        progress.progress(0.03, text="正在检查影像尺寸、坐标系和分辨率...")
        meta = validate_raster_shapes(paths)

        sprinklers = generate_sprinklers(cfg)
        sectors = sector_table(cfg)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("影像尺寸", f"{meta['width']} × {meta['height']}")
        c2.metric("上传总大小", f"{size_total:.1f} MB")
        c3.metric("喷头数量", f"{len(sprinklers)}")
        c4.metric("角度扇区", f"{len(sectors)}")

        if coordinate_mode == "使用 GeoTIFF 地理坐标，单位应为 m":
            st.info(f"参考影像 CRS：{meta['crs']}；像元大小：{meta['res']}。请确认坐标单位是 m。")
        else:
            st.info(f"本地坐标模式：像元大小按 {rs_cfg.pixel_size_m} m 计算，支轴中心坐标按本地米坐标填写。")

        with st.spinner("第一遍：分块抽样估算遥感指数分位数..."):
            stats_df = collect_index_percentiles(paths, rs_cfg, progress=progress)
        if stats_df.empty:
            st.error("无法从影像中提取有效植被像元。请检查反射率缩放系数、NDVI阈值、波段顺序和影像值范围。")
            st.stop()

        with st.spinner("第二遍：分块计算目标灌水量并聚合到喷头 × 角度扇区矩阵..."):
            polar_df = process_big_raster_to_polar(paths, cfg, rs_cfg, sprinklers, sectors, stats_df, progress=progress)

        if polar_df.empty or polar_df["target_mm"].sum() <= 0:
            st.warning("处方矩阵生成完成，但目标灌水量总和为 0。请检查支轴中心、覆盖半径、NDVI阈值和最大灌水量设置。")

        control_long, speed_df = select_sector_speed_and_duty(polar_df, cfg)
        control_wide = make_wide_matrix(control_long)
        progress.progress(1.0, text="完成。")

        elapsed = time.time() - t0
        st.success(f"大图处方图生成完成，用时 {elapsed:.1f} 秒。")

        run_summary = pd.DataFrame([
            {"item": "width", "value": meta["width"]},
            {"item": "height", "value": meta["height"]},
            {"item": "total_upload_mb", "value": round(size_total, 2)},
            {"item": "block_size", "value": rs_cfg.block_size},
            {"item": "sample_step", "value": rs_cfg.sample_step},
            {"item": "stats_sample_step", "value": rs_cfg.stats_sample_step},
            {"item": "sector_deg", "value": cfg.sector_deg},
            {"item": "sprinkler_count", "value": len(sprinklers)},
            {"item": "sector_count", "value": len(sectors)},
            {"item": "elapsed_seconds", "value": round(elapsed, 2)},
            {"item": "blue_used", "value": bool(paths.get("blue"))},
            {"item": "thermal_used", "value": bool(paths.get("thermal"))},
            {"item": "coordinate_mode", "value": coordinate_mode},
            {"item": "center_x", "value": cfg.center_x},
            {"item": "center_y", "value": cfg.center_y},
        ])

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["处方图预览", "指数统计", "速度处方", "喷头控制表", "控制矩阵"])
        with tab1:
            p1, p2 = st.columns(2)
            with p1:
                st.pyplot(plot_polar_target(polar_df))
            with p2:
                st.pyplot(plot_duty_heatmap(control_long))
            st.pyplot(plot_speed(speed_df))
        with tab2:
            st.dataframe(stats_df, use_container_width=True)
            st.dataframe(run_summary, use_container_width=True)
        with tab3:
            st.dataframe(speed_df, use_container_width=True)
        with tab4:
            st.dataframe(control_long, use_container_width=True)
        with tab5:
            st.dataframe(control_wide, use_container_width=True)

        files = {
            "01_sprinkler_parameters.csv": df_to_csv_bytes(sprinklers),
            "02_polar_target_depth.csv": df_to_csv_bytes(polar_df),
            "03_sector_speed_prescription.csv": df_to_csv_bytes(speed_df),
            "04_sprinkler_control_long.csv": df_to_csv_bytes(control_long),
            "05_duty_matrix_wide.csv": df_to_csv_bytes(control_wide),
            "06_index_percentile_stats.csv": df_to_csv_bytes(stats_df),
            "07_run_summary.csv": df_to_csv_bytes(run_summary),
        }
        zip_bytes = make_zip(files)
        st.download_button(
            label="下载全部处方结果 ZIP",
            data=zip_bytes,
            file_name="pivot_vri_large_upload_outputs.zip",
            mime="application/zip",
        )

        st.info(
            "本版本为 Streamlit 大图优化版。若后续要做正式商用软件，建议把上传改为阿里云 OSS 分片上传，"
            "后端 ECS Worker 从 OSS 读取或缓存 COG 文件，再执行同样的分块聚合算法。"
        )

    except Exception as e:
        st.exception(e)
    finally:
        # Streamlit 单次任务完成后清理临时文件，避免云端磁盘堆满。
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
