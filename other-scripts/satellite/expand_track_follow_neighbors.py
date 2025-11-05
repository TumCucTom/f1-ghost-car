#!/usr/bin/env python3
"""
Expand track_follow tiles by adding one-ring neighbors (+1 in each direction)
per tile in Web Mercator pixel space at the same zoom. Dedupe by center pixel
position, download missing images, append to metadata, and write URLs.

Run after hungaroring_track_follow_capture.py.
"""

import os
import csv
import time
from typing import Dict, Tuple, List, Set
import math

import requests

META_CSV = os.path.join("satellite_images", "track_follow", "metadata.csv")
IMG_DIR = os.path.join("satellite_images", "track_follow")
URLS_PATH = os.path.join("satellite_images", "track_follow", "track_follow_urls_neighbors.txt")

# Parameters should match capture settings
ZOOM = 20
SIZE = 640
SCALE = 2
MAPTYPE = "satellite"
API_KEY = "AIzaSyBrHdI6NkzcvxokQL5yr3f6NvOaq1FW_P4"
BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

# World-pixel side length covered by a SIZE tile (independent of SCALE)
WORLD_TILE = SIZE


def static_url(lat: float, lon: float) -> str:
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(ZOOM),
        "size": f"{SIZE}x{SIZE}",
        "scale": str(SCALE),
        "maptype": MAPTYPE,
        "key": API_KEY,
    }
    return BASE_URL + "?" + "&".join([f"{k}={v}" for k, v in params.items()])


def pixel_to_latlon(px: float, py: float, zoom: int) -> Tuple[float, float]:
    TILE_SIZE = 256
    world_size = TILE_SIZE * (1 << zoom)
    lon = px / world_size * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * py / world_size
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lon


def read_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_rows(path: str, rows: List[Dict[str, str]]):
    headers = ["idx", "track_idx", "lat", "lon", "px", "py", "left", "top", "right", "bottom", "filename"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    if not os.path.exists(META_CSV):
        print(f"Metadata not found: {META_CSV}")
        return

    rows = read_rows(META_CSV)

    # Build set of existing centers in world pixels (rounded to 1e-3 to avoid float noise)
    existing_keys: Set[Tuple[int, int]] = set()
    for r in rows:
        k = (int(round(float(r["px"]))), int(round(float(r["py"])))); existing_keys.add(k)

    # Neighbor offsets (8-neighborhood + orthogonals), step is WORLD_TILE in world pixels
    offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),           (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    new_rows: List[Dict[str, str]] = []
    next_idx = max(int(r["idx"]) for r in rows) + 1 if rows else 0

    for r in rows:
        cx = float(r["px"])
        cy = float(r["py"])
        for ox, oy in offsets:
            nx = cx + ox * WORLD_TILE
            ny = cy + oy * WORLD_TILE
            k = (int(round(nx)), int(round(ny)))
            if k in existing_keys:
                continue

            # Convert neighbor center to lat/lon
            lat, lon = pixel_to_latlon(nx, ny, ZOOM)

            # Compute world-pixel bounds
            half = WORLD_TILE / 2.0
            left = nx - half
            top = ny - half
            right = nx + half
            bottom = ny + half

            filename = f"track_{next_idx:04d}.png"
            new_rows.append({
                "idx": str(next_idx),
                "track_idx": r.get("track_idx", "-1"),
                "lat": f"{lat:.8f}",
                "lon": f"{lon:.8f}",
                "px": f"{nx:.2f}",
                "py": f"{ny:.2f}",
                "left": f"{left:.2f}",
                "top": f"{top:.2f}",
                "right": f"{right:.2f}",
                "bottom": f"{bottom:.2f}",
                "filename": filename,
            })
            existing_keys.add(k)
            next_idx += 1

    if not new_rows:
        print("No neighbor tiles to add.")
        return

    print(f"Adding {len(new_rows)} neighbor tiles...")

    # Append to CSV (keep simple: rewrite all rows with new appended)
    all_rows = rows + new_rows
    write_rows(META_CSV, all_rows)

    # Download newly added tiles and write URLs
    os.makedirs(IMG_DIR, exist_ok=True)
    with open(URLS_PATH, "w") as f:
        for r in new_rows:
            lat = float(r["lat"]); lon = float(r["lon"])  
            url = static_url(lat, lon)
            f.write(f"{r['filename']}: {url}\n")
            out_path = os.path.join(IMG_DIR, r["filename"])
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    with open(out_path, "wb") as imgf:
                        imgf.write(resp.content)
                else:
                    print(f"HTTP {resp.status_code} for {r['filename']}")
            except Exception as e:
                print(f"Error downloading {r['filename']}: {e}")
            time.sleep(0.1)

    print("Neighbor expansion complete.")


if __name__ == "__main__":
    main()


