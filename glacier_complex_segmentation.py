#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glacier_complex_segmentation.py

Automated glacier-complex segmentation based on DEM-derived topographic divides.

Main steps
----------
1. Clip DEM to glacier-complex polygon.
2. Hydrologically condition DEM.
3. Extract D8 flow direction, flow accumulation, and drainage basins.
4. Derive candidate divide pixels from basin-label boundaries.
5. Identify actual ridge pixels using:
   - positive/negative relief mask based on local TPI;
   - aspect-difference filter, delta_theta_min = 30 degrees by default.
6. Skeletonize and vectorize the filtered divide pixels.
7. Smooth divide lines using a midpoint smoothing method.
8. Segment glacier-complex polygons using the final divide lines.
9. Optionally attach nearest/intersecting RGI 7.0 IDs for consistency checking.

Author: Ruonan Li
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.mask import mask as rio_mask
from rasterio.transform import xy
from scipy.ndimage import distance_transform_edt, uniform_filter
from skimage.morphology import remove_small_objects, skeletonize
from shapely.geometry import (
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
    GeometryCollection,
    mapping,
    shape,
)
from shapely.ops import polygonize, unary_union

import whitebox


# -----------------------------
# Basic raster/vector utilities
# -----------------------------

def read_glacier_union(vector_path: Path, target_crs) -> tuple[gpd.GeoDataFrame, object]:
    """Read glacier-complex polygons and dissolve them into one geometry."""
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"No features found in {vector_path}")

    if gdf.crs is None:
        raise ValueError(
            f"{vector_path} has no CRS. Please define its CRS before running this script."
        )

    gdf = gdf.to_crs(target_crs)
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])

    if geom.is_empty:
        raise ValueError("Dissolved glacier geometry is empty.")

    return gdf, geom


def clip_dem_to_geometry(
    dem_path: Path,
    geometry,
    out_dem_path: Path,
    buffer_m: float = 500.0,
    nodata: float = -9999.0,
) -> tuple[np.ndarray, dict]:
    """
    Clip DEM to a buffered glacier-complex geometry.

    The buffer gives hydrological tools some terrain context around the glacier.
    """
    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError(f"{dem_path} has no CRS.")

        geom_for_clip = geometry.buffer(buffer_m) if buffer_m > 0 else geometry
        out_img, out_transform = rio_mask(
            src,
            [mapping(geom_for_clip)],
            crop=True,
            filled=True,
            nodata=nodata,
        )

        arr = out_img[0].astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[arr == nodata] = np.nan

        meta = src.meta.copy()
        meta.update(
            {
                "height": arr.shape[0],
                "width": arr.shape[1],
                "transform": out_transform,
                "dtype": "float32",
                "nodata": nodata,
                "compress": "lzw",
            }
        )

    arr_to_write = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    out_dem_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_dem_path, "w", **meta) as dst:
        dst.write(arr_to_write, 1)

    return arr, meta


def rasterize_glacier_mask(geometry, meta: dict, all_touched: bool = True) -> np.ndarray:
    """Rasterize glacier-complex geometry onto the clipped DEM grid."""
    mask = rasterize(
        [(mapping(geometry), 1)],
        out_shape=(meta["height"], meta["width"]),
        transform=meta["transform"],
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    return mask.astype(bool)


def write_uint8_raster(path: Path, arr: np.ndarray, meta: dict, nodata: int = 0) -> None:
    """Write a boolean/uint8 raster."""
    out_meta = meta.copy()
    out_meta.update(dtype="uint8", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(arr.astype("uint8"), 1)


# -----------------------------
# Hydrological conditioning
# -----------------------------

def run_whitebox_hydrology(
    clipped_dem: Path,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Hydrologically condition DEM and derive basins.

    WhiteboxTools steps:
    - BreachDepressions
    - D8Pointer
    - D8FlowAccumulation
    - Basins
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    breached = out_dir / "dem_breached.tif"
    d8_pointer = out_dir / "d8_pointer.tif"
    flow_acc = out_dir / "d8_flow_accumulation_cells.tif"
    basins = out_dir / "basins.tif"

    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(str(out_dir))
    wbt.verbose = False

    # Remove topographic depressions and small pits.
    wbt.breach_depressions(
        str(clipped_dem),
        str(breached),
        fill_pits=True,
    )

    # D8 local drainage direction.
    wbt.d8_pointer(
        str(breached),
        str(d8_pointer),
    )

    # Flow accumulation in cells. This is kept for diagnostics and optional thresholding.
    wbt.d8_flow_accumulation(
        str(breached),
        str(flow_acc),
        out_type="cells",
    )

    # Divide DEM domain into mutually exclusive drainage basins.
    wbt.basins(
        str(d8_pointer),
        str(basins),
    )

    return {
        "breached_dem": breached,
        "d8_pointer": d8_pointer,
        "flow_acc": flow_acc,
        "basins": basins,
    }


# -----------------------------
# Terrain filters
# -----------------------------

def nanmean_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Fast local mean that ignores NaN values."""
    valid = np.isfinite(arr)
    arr0 = np.where(valid, arr, 0.0).astype("float64")
    num = uniform_filter(arr0, size=size, mode="nearest")
    den = uniform_filter(valid.astype("float64"), size=size, mode="nearest")
    out = np.full(arr.shape, np.nan, dtype="float64")
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def fill_nans_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs using nearest valid cell; useful before computing slope/aspect."""
    valid = np.isfinite(arr)
    if valid.all():
        return arr.copy()

    if not valid.any():
        raise ValueError("DEM contains no valid cells.")

    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr[tuple(indices)]


def compute_tpi_positive_relief(
    dem: np.ndarray,
    tpi_radius_px: int = 5,
    tpi_threshold_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local topographic position index and positive-relief mask.

    Positive relief:
        TPI = elevation - local_mean_elevation
        positive_relief = TPI > tpi_threshold_m

    For a 12.5 m DEM, tpi_radius_px=5 means a window of 11 x 11 cells.
    """
    size = 2 * tpi_radius_px + 1
    local_mean = nanmean_filter(dem, size=size)
    tpi = dem - local_mean
    positive_relief = np.isfinite(tpi) & (tpi > tpi_threshold_m)
    return tpi, positive_relief


def compute_aspect_degrees(dem: np.ndarray, transform) -> np.ndarray:
    """
    Compute aspect in degrees clockwise from north.

    Output range: 0-360 degrees.
    """
    dem_filled = fill_nans_nearest(dem).astype("float64")

    cell_x = abs(transform.a)
    cell_y = abs(transform.e)

    dz_dy, dz_dx = np.gradient(dem_filled, cell_y, cell_x)

    # Aspect convention: 0 = north, 90 = east.
    aspect = (np.degrees(np.arctan2(dz_dx, -dz_dy)) + 360.0) % 360.0
    aspect[~np.isfinite(dem)] = np.nan

    return aspect


def shift_array(arr: np.ndarray, dr: int, dc: int, fill_value=np.nan) -> np.ndarray:
    """Shift array without wrap-around."""
    out = np.full(arr.shape, fill_value, dtype=arr.dtype)

    src_r0 = max(0, -dr)
    src_r1 = arr.shape[0] - max(0, dr)
    src_c0 = max(0, -dc)
    src_c1 = arr.shape[1] - max(0, dc)

    dst_r0 = max(0, dr)
    dst_r1 = arr.shape[0] - max(0, -dr)
    dst_c0 = max(0, dc)
    dst_c1 = arr.shape[1] - max(0, -dc)

    out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


def circular_angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute circular difference between two angle rasters."""
    diff = np.abs((a - b + 180.0) % 360.0 - 180.0)
    diff[~np.isfinite(a) | ~np.isfinite(b)] = np.nan
    return diff


def compute_two_sided_aspect_difference(aspect: np.ndarray) -> np.ndarray:
    """
    Estimate the maximum aspect difference across opposite sides of each cell.

    This approximates the CGI-style aspect-difference filter:
    two-sided ridge tops tend to show large aspect differences on opposite sides,
    whereas valley bottoms and uniform slopes are removed by combining this filter
    with the positive-relief mask.
    """
    opposite_pairs = [
        ((-1, 0), (1, 0)),     # north-south
        ((0, -1), (0, 1)),     # west-east
        ((-1, -1), (1, 1)),    # northwest-southeast
        ((-1, 1), (1, -1)),    # northeast-southwest
    ]

    max_diff = np.zeros(aspect.shape, dtype="float32")

    for (dr1, dc1), (dr2, dc2) in opposite_pairs:
        a1 = shift_array(aspect, dr1, dc1, fill_value=np.nan)
        a2 = shift_array(aspect, dr2, dc2, fill_value=np.nan)
        diff = circular_angle_difference(a1, a2)
        diff = np.where(np.isfinite(diff), diff, 0.0)
        max_diff = np.maximum(max_diff, diff.astype("float32"))

    max_diff[~np.isfinite(aspect)] = np.nan
    return max_diff


# -----------------------------
# Basin-boundary and ridge extraction
# -----------------------------

def basin_boundary_from_labels(
    labels: np.ndarray,
    glacier_mask: np.ndarray,
    min_contributing_pixels: int = 1,
) -> np.ndarray:
    """
    Extract boundary pixels between adjacent basin labels inside the glacier mask.

    min_contributing_pixels removes very small basins/noisy outlets.
    """
    labels = labels.astype("int64")
    valid = glacier_mask & (labels > 0)

    if min_contributing_pixels > 1:
        max_label = int(labels[valid].max()) if valid.any() else 0
        if max_label > 0:
            counts = np.bincount(labels[valid].ravel(), minlength=max_label + 1)
            keep = np.zeros(max_label + 1, dtype=bool)
            keep[counts >= min_contributing_pixels] = True

            keep_mask = np.zeros(labels.shape, dtype=bool)
            inside = (labels >= 0) & (labels <= max_label)
            keep_mask[inside] = keep[labels[inside]]
            valid = valid & keep_mask

    boundary = np.zeros(labels.shape, dtype=bool)

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted_labels = shift_array(labels, dr, dc, fill_value=0)
        shifted_valid = shift_array(valid.astype("uint8"), dr, dc, fill_value=0).astype(bool)

        boundary |= (
            valid
            & shifted_valid
            & (labels != shifted_labels)
            & (shifted_labels > 0)
        )

    return boundary


def extract_actual_ridge_mask(
    basin_boundary: np.ndarray,
    positive_relief: np.ndarray,
    aspect_difference: np.ndarray,
    delta_theta_min: float = 30.0,
    min_object_pixels: int = 8,
) -> np.ndarray:
    """
    Apply positive-relief and aspect-difference filtering to candidate divides.
    """
    ridge = (
        basin_boundary
        & positive_relief
        & np.isfinite(aspect_difference)
        & (aspect_difference >= delta_theta_min)
    )

    ridge = remove_small_objects(ridge.astype(bool), min_size=min_object_pixels)
    ridge = skeletonize(ridge)

    return ridge.astype(bool)


# -----------------------------
# Skeleton-to-vector conversion
# -----------------------------

NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def pixel_neighbors(pixel: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = pixel
    out = []
    for dr, dc in NEIGHBORS_8:
        q = (r + dr, c + dc)
        if q in pixels:
            out.append(q)
    return out


def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def pixel_path_to_line(
    path: list[tuple[int, int]],
    transform,
) -> LineString | None:
    if len(path) < 2:
        return None

    coords = [xy(transform, r, c, offset="center") for r, c in path]
    # Remove consecutive duplicate coordinates.
    clean = [coords[0]]
    for p in coords[1:]:
        if p != clean[-1]:
            clean.append(p)

    if len(clean) < 2:
        return None

    line = LineString(clean)
    return line if line.length > 0 else None


def skeleton_to_lines(
    skel: np.ndarray,
    transform,
    min_line_length_m: float = 50.0,
) -> list[LineString]:
    """
    Convert a 1-pixel skeleton raster to vector LineStrings.

    This function traces paths from endpoints/junctions and also handles closed loops.
    """
    pixels = {tuple(p) for p in np.argwhere(skel)}
    if not pixels:
        return []

    degree = {p: len(pixel_neighbors(p, pixels)) for p in pixels}
    starts = [p for p, d in degree.items() if d != 2]

    visited_edges = set()
    lines: list[LineString] = []

    def add_line(path):
        line = pixel_path_to_line(path, transform)
        if line is not None and line.length >= min_line_length_m:
            lines.append(line)

    # Trace from endpoints and junctions.
    for start in starts:
        for nb in pixel_neighbors(start, pixels):
            ek = edge_key(start, nb)
            if ek in visited_edges:
                continue

            path = [start, nb]
            visited_edges.add(ek)
            prev, cur = start, nb

            while degree[cur] == 2:
                nbs = pixel_neighbors(cur, pixels)
                nxt = nbs[0] if nbs[1] == prev else nbs[1]
                ek = edge_key(cur, nxt)
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                path.append(nxt)
                prev, cur = cur, nxt

            add_line(path)

    # Trace remaining closed loops.
    for p in list(pixels):
        for nb in pixel_neighbors(p, pixels):
            ek = edge_key(p, nb)
            if ek in visited_edges:
                continue

            path = [p]
            prev = None
            cur = p

            while True:
                nbs = [q for q in pixel_neighbors(cur, pixels) if q != prev]
                if not nbs:
                    break

                nxt = nbs[0]
                ek = edge_key(cur, nxt)
                if ek in visited_edges:
                    break

                visited_edges.add(ek)
                path.append(nxt)
                prev, cur = cur, nxt

                if cur == p:
                    break

            add_line(path)

    return lines


# -----------------------------
# Midpoint smoothing and segmentation
# -----------------------------

def midpoint_smooth_line(line: LineString, iterations: int = 2) -> LineString:
    """
    Smooth a line by repeatedly replacing each internal vertex with the midpoint
    of its two neighboring vertices.

    Endpoints are preserved so that divides remain anchored to glacier boundaries
    as much as possible.
    """
    if line.is_empty or len(line.coords) < 3 or iterations <= 0:
        return line

    coords = list(line.coords)

    for _ in range(iterations):
        if len(coords) < 3:
            break

        new_coords = [coords[0]]
        for i in range(1, len(coords) - 1):
            x = 0.5 * (coords[i - 1][0] + coords[i + 1][0])
            y = 0.5 * (coords[i - 1][1] + coords[i + 1][1])
            new_coords.append((x, y))
        new_coords.append(coords[-1])

        # Drop accidental consecutive duplicates.
        clean = [new_coords[0]]
        for p in new_coords[1:]:
            if p != clean[-1]:
                clean.append(p)

        coords = clean

    if len(coords) < 2:
        return line

    return LineString(coords)


def iter_lines(geom) -> Iterable[LineString]:
    """Flatten LineString/MultiLineString/GeometryCollection."""
    if geom is None or geom.is_empty:
        return

    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            yield from iter_lines(g)
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from iter_lines(g)


def split_polygon_by_divides(
    glacier_geom,
    divide_lines: list[LineString],
    min_segment_area_m2: float = 1000.0,
) -> list[Polygon]:
    """
    Segment glacier-complex geometry using divide lines.

    This polygonizes the union of:
    - glacier outer boundary;
    - internal divide lines.

    If divide lines do not fully connect to the glacier boundary, polygonize may
    return only the original glacier polygon. In that case, use the raster basin
    polygonization fallback below.
    """
    if not divide_lines:
        return []

    linework = unary_union([glacier_geom.boundary, unary_union(divide_lines)])
    pieces = []

    for poly in polygonize(linework):
        inter = poly.intersection(glacier_geom)
        if inter.is_empty:
            continue

        if isinstance(inter, Polygon):
            geoms = [inter]
        elif isinstance(inter, MultiPolygon):
            geoms = list(inter.geoms)
        else:
            continue

        for g in geoms:
            if g.area >= min_segment_area_m2 and g.representative_point().within(glacier_geom):
                pieces.append(g)

    return pieces


def fallback_segments_from_basin_labels(
    basin_labels: np.ndarray,
    glacier_mask: np.ndarray,
    transform,
    glacier_geom,
    min_segment_area_m2: float = 1000.0,
) -> list[Polygon]:
    """
    Fallback segmentation by polygonizing basin labels inside the glacier complex.

    This is useful when filtered ridge lines are too short to split polygons.
    """
    arr = np.where(glacier_mask, basin_labels, 0).astype("int32")

    segments = []
    for geom_json, val in shapes(arr, mask=(arr > 0), transform=transform):
        if int(val) <= 0:
            continue

        poly = shape(geom_json).intersection(glacier_geom)
        if poly.is_empty:
            continue

        if isinstance(poly, Polygon):
            geoms = [poly]
        elif isinstance(poly, MultiPolygon):
            geoms = list(poly.geoms)
        else:
            continue

        for g in geoms:
            if g.area >= min_segment_area_m2:
                segments.append(g)

    return segments


def attach_rgi7_id(
    segments_gdf: gpd.GeoDataFrame,
    rgi7_path: Path | None,
) -> gpd.GeoDataFrame:
    """
    Optionally attach the most-overlapping RGI 7.0 glacier ID to each segment.

    This does not use RGI to force the segmentation. It only adds a reference ID
    for cross-inventory consistency checking.
    """
    if rgi7_path is None:
        return segments_gdf

    rgi = gpd.read_file(rgi7_path).to_crs(segments_gdf.crs)

    candidate_id_cols = [
        "rgi_id", "RGIId", "RGI_ID", "rgiid",
        "rgi7_id", "RGI7_ID", "glac_id", "GLAC_ID",
    ]
    id_col = next((c for c in candidate_id_cols if c in rgi.columns), None)

    if id_col is None:
        print("Warning: No recognizable RGI ID field found. Skipping RGI ID attachment.")
        return segments_gdf

    seg = segments_gdf.copy()
    seg["seg_ix"] = seg.index

    inter = gpd.overlay(
        seg[["seg_ix", "geometry"]],
        rgi[[id_col, "geometry"]],
        how="intersection",
    )

    if inter.empty:
        segments_gdf["rgi7_id"] = None
        return segments_gdf

    inter["inter_area"] = inter.geometry.area
    best = inter.sort_values("inter_area").groupby("seg_ix").tail(1)

    lookup = best.set_index("seg_ix")[id_col].to_dict()
    segments_gdf["rgi7_id"] = segments_gdf.index.map(lookup)

    return segments_gdf


# -----------------------------
# Main workflow
# -----------------------------

def segment_glacier_complex(
    dem_path: Path,
    glacier_complex_path: Path,
    out_dir: Path,
    rgi7_path: Path | None = None,
    clip_buffer_m: float = 500.0,
    min_contributing_area_m2: float = 5000.0,
    delta_theta_min: float = 30.0,
    tpi_radius_px: int = 5,
    tpi_threshold_m: float = 0.0,
    min_ridge_object_pixels: int = 8,
    min_line_length_m: float = 50.0,
    smooth_iterations: int = 2,
    min_segment_area_m2: float = 1000.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "intermediate"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        if dem_crs is None:
            raise ValueError("DEM has no CRS.")

        if not dem_crs.is_projected:
            raise ValueError(
                "DEM CRS should be projected in metres before running this workflow. "
                "Reproject DEM and glacier polygons first."
            )

    glacier_gdf, glacier_geom = read_glacier_union(glacier_complex_path, dem_crs)

    clipped_dem_path = tmp_dir / "dem_clipped.tif"
    dem_arr, meta = clip_dem_to_geometry(
        dem_path=dem_path,
        geometry=glacier_geom,
        out_dem_path=clipped_dem_path,
        buffer_m=clip_buffer_m,
    )

    transform = meta["transform"]
    pixel_area = abs(transform.a * transform.e)
    min_contributing_pixels = max(1, int(round(min_contributing_area_m2 / pixel_area)))

    glacier_mask = rasterize_glacier_mask(glacier_geom, meta, all_touched=True)
    write_uint8_raster(tmp_dir / "glacier_mask.tif", glacier_mask, meta)

    # Hydrological processing.
    hydro_paths = run_whitebox_hydrology(clipped_dem_path, tmp_dir)

    with rasterio.open(hydro_paths["basins"]) as src:
        basin_labels = src.read(1).astype("int64")
        basin_labels[basin_labels < 0] = 0

    # Candidate drainage divides from basin boundaries.
    candidate_boundary = basin_boundary_from_labels(
        labels=basin_labels,
        glacier_mask=glacier_mask,
        min_contributing_pixels=min_contributing_pixels,
    )
    write_uint8_raster(tmp_dir / "candidate_basin_boundaries.tif", candidate_boundary, meta)

    # Relief and aspect filters.
    tpi, positive_relief = compute_tpi_positive_relief(
        dem=dem_arr,
        tpi_radius_px=tpi_radius_px,
        tpi_threshold_m=tpi_threshold_m,
    )
    aspect = compute_aspect_degrees(dem_arr, transform)
    aspect_diff = compute_two_sided_aspect_difference(aspect)

    actual_ridge = extract_actual_ridge_mask(
        basin_boundary=candidate_boundary,
        positive_relief=positive_relief,
        aspect_difference=aspect_diff,
        delta_theta_min=delta_theta_min,
        min_object_pixels=min_ridge_object_pixels,
    )

    write_uint8_raster(tmp_dir / "actual_ridge_skeleton.tif", actual_ridge, meta)

    # Vectorize skeleton ridges.
    raw_lines = skeleton_to_lines(
        skel=actual_ridge,
        transform=transform,
        min_line_length_m=min_line_length_m,
    )

    smoothed_lines = [
        midpoint_smooth_line(line, iterations=smooth_iterations)
        for line in raw_lines
        if line.length >= min_line_length_m
    ]

    divides_gdf = gpd.GeoDataFrame(
        {
            "divide_id": list(range(1, len(smoothed_lines) + 1)),
            "length_m": [line.length for line in smoothed_lines],
            "delta_min": delta_theta_min,
            "geometry": smoothed_lines,
        },
        crs=dem_crs,
    )

    divides_path = out_dir / "topographic_divides.gpkg"
    divides_gdf.to_file(divides_path, layer="divides", driver="GPKG")

    # Segment glacier complex by divide lines.
    segments = split_polygon_by_divides(
        glacier_geom=glacier_geom,
        divide_lines=smoothed_lines,
        min_segment_area_m2=min_segment_area_m2,
    )

    # Fallback: polygonize basin labels if vector divides do not split enough.
    if len(segments) <= 1:
        print(
            "Warning: filtered divide lines did not sufficiently split the complex. "
            "Using basin-label polygonization as fallback."
        )
        segments = fallback_segments_from_basin_labels(
            basin_labels=basin_labels,
            glacier_mask=glacier_mask,
            transform=transform,
            glacier_geom=glacier_geom,
            min_segment_area_m2=min_segment_area_m2,
        )

    segments_gdf = gpd.GeoDataFrame(
        {
            "segment_id": list(range(1, len(segments) + 1)),
            "area_m2": [g.area for g in segments],
            "area_km2": [g.area / 1e6 for g in segments],
            "min_acc_m2": min_contributing_area_m2,
            "dtheta_min": delta_theta_min,
            "geometry": segments,
        },
        crs=dem_crs,
    )

    segments_gdf = attach_rgi7_id(segments_gdf, rgi7_path)

    segments_path = out_dir / "glacier_complex_segments.gpkg"
    segments_gdf.to_file(segments_path, layer="segments", driver="GPKG")

    print("Finished glacier-complex segmentation.")
    print(f"Divides:  {divides_path}")
    print(f"Segments: {segments_path}")
    print(f"Intermediate rasters: {tmp_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated glacier-complex segmentation using DEM-derived topographic divides."
    )

    parser.add_argument("--dem", required=True, type=Path, help="Input DEM GeoTIFF, projected CRS in metres.")
    parser.add_argument("--glacier-complex", required=True, type=Path, help="Input glacier-complex polygon file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")

    parser.add_argument("--rgi7", default=None, type=Path, help="Optional RGI 7.0 polygon file for ID attachment.")

    parser.add_argument("--clip-buffer-m", default=500.0, type=float)
    parser.add_argument("--min-contributing-area-m2", default=5000.0, type=float)
    parser.add_argument("--delta-theta-min", default=30.0, type=float)
    parser.add_argument("--tpi-radius-px", default=5, type=int)
    parser.add_argument("--tpi-threshold-m", default=0.0, type=float)
    parser.add_argument("--min-ridge-object-pixels", default=8, type=int)
    parser.add_argument("--min-line-length-m", default=50.0, type=float)
    parser.add_argument("--smooth-iterations", default=2, type=int)
    parser.add_argument("--min-segment-area-m2", default=1000.0, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    segment_glacier_complex(
        dem_path=args.dem,
        glacier_complex_path=args.glacier_complex,
        out_dir=args.out_dir,
        rgi7_path=args.rgi7,
        clip_buffer_m=args.clip_buffer_m,
        min_contributing_area_m2=args.min_contributing_area_m2,
        delta_theta_min=args.delta_theta_min,
        tpi_radius_px=args.tpi_radius_px,
        tpi_threshold_m=args.tpi_threshold_m,
        min_ridge_object_pixels=args.min_ridge_object_pixels,
        min_line_length_m=args.min_line_length_m,
        smooth_iterations=args.smooth_iterations,
        min_segment_area_m2=args.min_segment_area_m2,
    )