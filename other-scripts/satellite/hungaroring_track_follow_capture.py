#!/usr/bin/env python3
"""
Hungaroring Track-Follow Satellite Imagery Capture

This script follows the Hungaroring track centerline and downloads a sequence of
adjacent, non-overlapping Google Static Map satellite images that together cover
the full track. It starts from the given start-line coordinate and proceeds along
the track, ensuring each image's footprint (in Web Mercator pixel space) does not
overlap with previously captured images.

Outputs:
- satellite_images/track_follow/track_XXXX.png  # ordered along track
- satellite_images/track_follow/metadata.csv    # image metadata and bounds
- satellite_images/track_follow_urls.txt        # reproducible Static Maps URLs

Usage:
    python hungaroring_track_follow_capture.py
"""

import os
import math
import time
import csv
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

# --- Configuration ---
API_KEY = "AIzaSyBrHdI6NkzcvxokQL5yr3f6NvOaq1FW_P4"
BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

ZOOM = 20
SIZE = 640               # pixels (width == height), coverage computed at scale=1 world pixels
SCALE = 2                # returns higher-res imagery for the same geographic footprint
MAPTYPE = "satellite"

# Start-line anchor from the user's reference URL
START_LAT = 47.578885
START_LON = 19.248433

# Track CSV with centerline in local meters (x_m east, y_m north)
TRACK_CSV = os.path.join("track_data", "Budapest_minimum_curvature.csv")

# Margin around track in meters for conversion accuracy is handled by spacing logic

# Output directory
OUT_DIR = os.path.join("satellite_images", "track_follow")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Web Mercator helpers (scale-independent world pixel space) ---
TILE_SIZE = 256

def latlon_to_pixel(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    world_size = TILE_SIZE * (1 << zoom)
    x = (lon + 180.0) / 360.0 * world_size
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world_size
    return x, y

def pixel_to_latlon(x: float, y: float, zoom: int) -> Tuple[float, float]:
    world_size = TILE_SIZE * (1 << zoom)
    lon = x / world_size * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / world_size
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lon

# --- Local meters <-> lat/lon conversion (small-area approximation) ---
METERS_PER_DEG_LAT = 111_320.0

def meters_to_latlon(x_m: float, y_m: float, anchor_lat: float, anchor_lon: float) -> Tuple[float, float]:
    lat = anchor_lat + (y_m / METERS_PER_DEG_LAT)
    lon = anchor_lon + (x_m / (METERS_PER_DEG_LAT * math.cos(math.radians(anchor_lat))))
    return lat, lon

# --- Geometry helpers ---

def rect_from_center_px(cx: float, cy: float, size: int) -> Tuple[float, float, float, float]:
    half = size / 2.0
    return (cx - half, cy - half, cx + half, cy + half)  # left, top, right, bottom

def rects_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def point_in_rect(px: float, py: float, r: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = r
    return (x1 <= px <= x2) and (y1 <= py <= y2)

# --- Downloader ---

def static_map_url(lat: float, lon: float) -> str:
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(ZOOM),
        "size": f"{SIZE}x{SIZE}",
        "scale": str(SCALE),
        "maptype": MAPTYPE,
        "key": API_KEY,
    }
    return BASE_URL + "?" + "&".join([f"{k}={v}" for k, v in params.items()])

def download_image(lat: float, lon: float, out_path: str) -> bool:
    url = static_map_url(lat, lon)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
        else:
            print(f"HTTP {r.status_code} for {out_path}")
            return False
    except Exception as e:
        print(f"Error downloading {out_path}: {e}")
        return False

# --- Main logic ---

def main():
    print("Track-follow satellite capture (non-overlapping)")

    # Load track polyline (meters)
    if not os.path.exists(TRACK_CSV):
        print(f"Track CSV not found: {TRACK_CSV}")
        return
    df = pd.read_csv(TRACK_CSV, comment="#", names=["x_m", "y_m"])  # minimum curvature has 2 cols
    x_m = df["x_m"].values
    y_m = df["y_m"].values

    # Convert entire track to lat/lon using the start-line anchor as origin
    latlon = [meters_to_latlon(x, y, START_LAT, START_LON) for x, y in zip(x_m, y_m)]

    # Convert to pixel coordinates at the chosen zoom
    pixels = np.array([latlon_to_pixel(lat, lon, ZOOM) for lat, lon in latlon], dtype=float)

    # Find the track point closest (in pixel space) to the given start-line lat/lon
    sx, sy = latlon_to_pixel(START_LAT, START_LON, ZOOM)
    dists = np.hypot(pixels[:, 0] - sx, pixels[:, 1] - sy)
    start_idx = int(np.argmin(dists))
    print(f"Start index on track: {start_idx} (pixel distance {dists[start_idx]:.2f})")

    # Iterate along the track, placing non-overlapping image rectangles that include new track points
    visited_rects: List[Tuple[float, float, float, float]] = []
    covered = np.zeros(len(pixels), dtype=bool)

    # We will walk forward from start_idx through all points (wrap around once)
    order = list(range(start_idx, len(pixels))) + list(range(0, start_idx))

    images_meta: List[Dict[str, object]] = []

    # Helper to check if a candidate rect overlaps any previous rect
    def overlaps_any(rect: Tuple[float, float, float, float]) -> bool:
        for r in visited_rects:
            if rects_intersect(rect, r):
                return True
        return False

    # Helper to mark covered track points for a rect
    def mark_covered(rect: Tuple[float, float, float, float]):
        x1, y1, x2, y2 = rect
        in_x = (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2)
        in_y = (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
        covered[:] = covered | (in_x & in_y)

    # Place the very first image centered exactly on the given start-line coordinate
    first_rect = rect_from_center_px(sx, sy, SIZE)
    visited_rects.append(first_rect)
    f_lat, f_lon = START_LAT, START_LON
    images_meta.append({
        "idx": 0,
        "track_idx": start_idx,
        "lat": f_lat,
        "lon": f_lon,
        "px": sx,
        "py": sy,
        "left": first_rect[0],
        "top": first_rect[1],
        "right": first_rect[2],
        "bottom": first_rect[3],
        "filename": f"track_{0:04d}.png",
        "url": static_map_url(f_lat, f_lon),
    })
    mark_covered(first_rect)

    # Now walk forward and add images whenever the image rectangle would NOT overlap earlier ones
    img_count = 1
    for ti in order:
        cx, cy = pixels[ti]
        rect = rect_from_center_px(cx, cy, SIZE)
        if overlaps_any(rect):
            continue  # would overlap; keep walking forward

        # Only add if the rect contains some not-yet-covered track points
        x1, y1, x2, y2 = rect
        in_rect = ((pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2))
        if not np.any(~covered & in_rect):
            continue

        lat, lon = latlon[ti]
        images_meta.append({
            "idx": img_count,
            "track_idx": ti,
            "lat": lat,
            "lon": lon,
            "px": cx,
            "py": cy,
            "left": x1,
            "top": y1,
            "right": x2,
            "bottom": y2,
            "filename": f"track_{img_count:04d}.png",
            "url": static_map_url(lat, lon),
        })
        visited_rects.append(rect)
        mark_covered(rect)
        img_count += 1

        # Stop if all track points are covered
        if covered.all():
            break

    print(f"Planned {len(images_meta)} non-overlapping images along the track.")

    # Write URLs file and metadata CSV
    urls_path = os.path.join(OUT_DIR, "track_follow_urls.txt")
    with open(urls_path, "w") as f:
        for m in images_meta:
            f.write(f"{m['filename']}: {m['url']}\n")
    print(f"Saved URLs: {urls_path}")

    meta_csv = os.path.join(OUT_DIR, "metadata.csv")
    with open(meta_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "track_idx", "lat", "lon", "px", "py", "left", "top", "right", "bottom", "filename"])
        for m in images_meta:
            writer.writerow([m["idx"], m["track_idx"], f"{m['lat']:.8f}", f"{m['lon']:.8f}", f"{m['px']:.2f}", f"{m['py']:.2f}", f"{m['left']:.2f}", f"{m['top']:.2f}", f"{m['right']:.2f}", f"{m['bottom']:.2f}", m["filename"]])
    print(f"Saved metadata: {meta_csv}")

    # Download in-order
    print("Downloading images...")
    ok = 0
    for m in images_meta:
        out_path = os.path.join(OUT_DIR, m["filename"])
        if download_image(m["lat"], m["lon"], out_path):
            ok += 1
            time.sleep(0.1)
    print(f"Downloaded {ok}/{len(images_meta)} images to {OUT_DIR}")

if __name__ == "__main__":
    main()


