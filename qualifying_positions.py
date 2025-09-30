#!/usr/bin/env python3
"""
F1 Track Position Data Extraction Script

This script retrieves track XY positions for drivers at every time step during qualifying
using the Fast F1 API. It can compare two drivers' positions throughout the session.

Usage:
    python qualifying_positions.py --year 2024 --race "Monaco" --driver1 "VER" --driver2 "HAM"
    python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --export
"""

import argparse
import sys
import os
import fastf1
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


def load_track_data(track_name: str) -> Optional[pd.DataFrame]:
    """Load track centerline and edge data from CSV file."""
    try:
        csv_path = f"{track_name}.csv"
        if os.path.exists(csv_path):
            # Read CSV, skipping the header comment line
            df = pd.read_csv(csv_path, comment='#', names=['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
            return df
        else:
            print(f"Track data file not found: {csv_path}")
            return None
    except Exception as e:
        print(f"Error loading track data: {e}")
        return None


def create_road_from_track_data(track_df: pd.DataFrame, data1_track: pd.DataFrame, data2_track: pd.DataFrame, ax=None):
    """Create road surface from track centerline and edge data, aligned with driver positions."""
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize
    
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt
    
    # Get centerline coordinates
    center_x = track_df['x_m'].values
    center_y = track_df['y_m'].values
    
    # Get edge distances
    right_dist = track_df['w_tr_right_m'].values
    left_dist = track_df['w_tr_left_m'].values
    
    # Combine driver data to find the racing line
    driver_x = np.concatenate([data1_track['X'].values, data2_track['X'].values])
    driver_y = np.concatenate([data1_track['Y'].values, data2_track['Y'].values])
    
    # Find the best alignment between track data and driver data
    def alignment_error(params):
        scale, tx, ty = params
        
        # Scale and translate the track data
        scaled_center_x = center_x * scale + tx
        scaled_center_y = center_y * scale + ty
        
        # Find the best translation by minimizing distance to driver points
        track_points = np.column_stack([scaled_center_x, scaled_center_y])
        driver_points = np.column_stack([driver_x, driver_y])
        
        # Calculate minimum distances from each driver point to track
        distances = cdist(driver_points, track_points)
        min_distances = np.min(distances, axis=1)
        
        # Add penalty for points that are too far from track
        penalty = np.sum(np.maximum(0, min_distances - 30))  # Reduced penalty threshold
        
        return np.mean(min_distances) + penalty * 0.2
    
    # Try multiple starting points for better optimization
    best_result = None
    best_error = float('inf')
    
    # Test different starting points
    starting_points = [
        [9.717, -1000, -272],  # Previous result
        [8.875, 0, 0],         # Original scale
        [10.0, -500, -100],    # Larger scale, moderate shift
        [9.0, -1500, -400],    # Smaller scale, larger shift
        [11.0, -800, -200],    # Even larger scale
    ]
    
    for start_point in starting_points:
        try:
            result = minimize(alignment_error, start_point, 
                             bounds=[(5.0, 15.0), (-2000, 2000), (-2000, 2000)],
                             method='L-BFGS-B')
            
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
        except:
            continue
    
    if best_result is None:
        # Fallback to simple optimization
        best_result = minimize(alignment_error, [9.717, -1000, -272], 
                              bounds=[(5.0, 15.0), (-2000, 2000), (-2000, 2000)],
                              method='L-BFGS-B')
    
    optimal_scale, optimal_tx, optimal_ty = best_result.x
    
    # Apply optimal transformation
    aligned_center_x = center_x * optimal_scale + optimal_tx
    aligned_center_y = center_y * optimal_scale + optimal_ty
    
    # Scale the edge distances as well
    scaled_right_dist = right_dist * optimal_scale
    scaled_left_dist = left_dist * optimal_scale
    
    # Calculate direction vectors along the track
    dx = np.gradient(aligned_center_x)
    dy = np.gradient(aligned_center_y)
    
    # Normalize direction vectors
    length = np.sqrt(dx**2 + dy**2)
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Calculate perpendicular vectors (rotated 90 degrees)
    perp_x = -dy_norm
    perp_y = dx_norm
    
    # Create left and right edge points
    left_x = aligned_center_x + perp_x * scaled_left_dist
    left_y = aligned_center_y + perp_y * scaled_left_dist
    right_x = aligned_center_x - perp_x * scaled_right_dist
    right_y = aligned_center_y - perp_y * scaled_right_dist
    
    # Create road polygon by combining left and right edges
    road_x = np.concatenate([left_x, right_x[::-1]])
    road_y = np.concatenate([left_y, right_y[::-1]])
    
    # Plot the road as a filled polygon
    ax.fill(road_x, road_y, color='#808080', alpha=0.6, zorder=1, label='Road Surface')
    
    # Add road outline
    ax.plot(road_x, road_y, color='#606060', linewidth=2, alpha=0.8, zorder=2)
    
    # Calculate final alignment quality
    track_points = np.column_stack([aligned_center_x, aligned_center_y])
    driver_points = np.column_stack([driver_x, driver_y])
    distances = cdist(driver_points, track_points)
    min_distances = np.min(distances, axis=1)
    avg_distance = np.mean(min_distances)
    max_distance = np.max(min_distances)
    
    print(f"Track aligned with scale: {optimal_scale:.3f}, translation: ({optimal_tx:.1f}, {optimal_ty:.1f})")
    print(f"Alignment quality - Avg distance: {avg_distance:.1f}m, Max distance: {max_distance:.1f}m")
    
    return True


def load_car_image(driver_code: str, size: int = 30, rotation: float = 0, maintain_aspect: bool = False) -> Optional[Image.Image]:
    """Load car image for a driver with optional rotation and aspect ratio preservation."""
    try:
        image_path = f"assets/cars/{driver_code.lower()}.png"
        if os.path.exists(image_path):
            img = Image.open(image_path)
            original_width, original_height = img.size
            
            if maintain_aspect:
                # Maintain aspect ratio - use size as the longer dimension
                aspect_ratio = original_width / original_height
                if aspect_ratio > 1:  # Wider than tall
                    new_width = size
                    new_height = int(size / aspect_ratio)
                else:  # Taller than wide or square
                    new_height = size
                    new_width = int(size * aspect_ratio)
                
                # Use even higher resolution for better quality when maintaining aspect ratio
                high_res_multiplier = 8  # 8x resolution for better quality
                high_res_width = new_width * high_res_multiplier
                high_res_height = new_height * high_res_multiplier
                
                img = img.resize((high_res_width, high_res_height), Image.Resampling.LANCZOS)
                
                # Rotate the image if needed (at high resolution)
                if rotation != 0:
                    img = img.rotate(rotation, expand=True, fillcolor=(255, 255, 255, 0))
                    # After rotation, we need to recalculate the final size to maintain aspect ratio
                    # Get the rotated dimensions
                    rotated_width, rotated_height = img.size
                    # Calculate the scale factor to fit within our target size
                    scale_factor = min(new_width / rotated_width, new_height / rotated_height)
                    final_width = int(rotated_width * scale_factor)
                    final_height = int(rotated_height * scale_factor)
                    # Resize to final dimensions
                    img = img.resize((final_width, final_height), Image.Resampling.LANCZOS)
                else:
                    # Scale down to final size with high quality
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                # Original square approach
                high_res_size = size * 4  # 4x resolution for better quality
                img = img.resize((high_res_size, high_res_size), Image.Resampling.LANCZOS)
                
                # Rotate the image if needed (at high resolution)
                if rotation != 0:
                    img = img.rotate(rotation, expand=True, fillcolor=(255, 255, 255, 0))
                
                # Scale down to final size with high quality
                img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            return img
        else:
            print(f"Car image not found: {image_path}")
            return None
    except Exception as e:
        print(f"Error loading car image for {driver_code}: {e}")
        return None


def calculate_direction(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the direction angle in degrees from point 1 to point 2."""
    import math
    dx = x2 - x1
    dy = y2 - y1
    # Calculate angle in radians, then convert to degrees
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def get_track_position_data(year: int, race: str, driver1: str, driver2: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Get track XY position data for two drivers throughout the qualifying session.
    
    Args:
        year: The year of the race
        race: The name of the Grand Prix
        driver1: Three-letter driver code for first driver
        driver2: Three-letter driver code for second driver
    
    Returns:
        Tuple of (driver1_position_data, driver2_position_data) or (None, None) if error
    """
    try:
        # Create cache directory if it doesn't exist
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"Created cache directory: {cache_dir}")
        
        # Enable caching for better performance
        fastf1.Cache.enable_cache(cache_dir)
        
        # Load the qualifying session
        print(f"Loading qualifying session for {year} {race}...")
        session = fastf1.get_session(year, race, 'Q')
        session.load()
        
        # Debug: Show available drivers and their codes
        results = session.results
        print(f"Available drivers in session:")
        for _, driver in results.iterrows():
            print(f"  {driver['Abbreviation']} ({driver['FullName']}) - Position: {driver['Position']}")
        
        # Get position data for both drivers
        position_data1 = get_driver_position_data(session, driver1)
        position_data2 = get_driver_position_data(session, driver2)
        
        return position_data1, position_data2
        
    except Exception as e:
        print(f"Error loading session data: {e}")
        return None, None


def get_driver_position_data(session, driver_code: str) -> Optional[pd.DataFrame]:
    """
    Get track position data (X, Y coordinates) for a specific driver throughout the session.
    
    Args:
        session: FastF1 session object
        driver_code: Three-letter driver code
    
    Returns:
        DataFrame with position data or None if not found
    """
    try:
        # Get the session results to find the driver
        results = session.results
        driver_result = results.loc[results['Abbreviation'] == driver_code]
        
        if driver_result.empty:
            print(f"Driver {driver_code} not found in session")
            return None
        
        # Get the driver number
        driver_number = driver_result.iloc[0]['DriverNumber']
        
        # Get position data for this driver
        position_data = session.pos_data[driver_number]
        
        if position_data.empty:
            print(f"No position data found for driver {driver_code}")
            return None
        
        # Add driver information to the data
        position_data = position_data.copy()
        position_data['Driver'] = driver_code
        position_data['DriverNumber'] = driver_number
        
        print(f"Retrieved {len(position_data)} position data points for {driver_code}")
        
        return position_data
        
    except Exception as e:
        print(f"Error getting position data for driver {driver_code}: {e}")
        return None


def analyze_position_data(driver1: str, data1: Optional[pd.DataFrame], 
                         driver2: str, data2: Optional[pd.DataFrame]) -> None:
    """
    Analyze and display track position data for two drivers.
    
    Args:
        driver1: Name/code of first driver
        data1: Position data DataFrame for first driver
        driver2: Name/code of second driver
        data2: Position data DataFrame for second driver
    """
    print("\n" + "="*60)
    print("TRACK POSITION DATA ANALYSIS")
    print("="*60)
    
    if data1 is None or data2 is None:
        print("Unable to retrieve position data for analysis.")
        return
    
    # Display basic statistics
    print(f"\n{driver1} Position Data:")
    print(f"  Total data points: {len(data1)}")
    print(f"  Time range: {data1.index[0]} to {data1.index[-1]}")
    print(f"  X coordinate range: {data1['X'].min():.2f} to {data1['X'].max():.2f}")
    print(f"  Y coordinate range: {data1['Y'].min():.2f} to {data1['Y'].max():.2f}")
    
    print(f"\n{driver2} Position Data:")
    print(f"  Total data points: {len(data2)}")
    print(f"  Time range: {data2.index[0]} to {data2.index[-1]}")
    print(f"  X coordinate range: {data2['X'].min():.2f} to {data2['X'].max():.2f}")
    print(f"  Y coordinate range: {data2['Y'].min():.2f} to {data2['Y'].max():.2f}")
    
    # Show sample data
    print(f"\nSample data for {driver1} (first 5 points):")
    print(data1[['X', 'Y', 'Status']].head())
    
    print(f"\nSample data for {driver2} (first 5 points):")
    print(data2[['X', 'Y', 'Status']].head())


def export_position_data(driver1: str, data1: Optional[pd.DataFrame], 
                        driver2: str, data2: Optional[pd.DataFrame], 
                        year: int, race: str) -> None:
    """
    Export position data to CSV files.
    
    Args:
        driver1: Name/code of first driver
        data1: Position data DataFrame for first driver
        driver2: Name/code of second driver
        data2: Position data DataFrame for second driver
        year: Year of the race
        race: Name of the race
    """
    if data1 is None or data2 is None:
        print("Cannot export data - missing position data.")
        return
    
    # Create export directory
    export_dir = f"position_data_{year}_{race.replace(' ', '_')}"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"Created export directory: {export_dir}")
    
    # Export individual driver data
    filename1 = f"{export_dir}/{driver1}_position_data.csv"
    data1.to_csv(filename1)
    print(f"Exported {driver1} data to: {filename1}")
    
    filename2 = f"{export_dir}/{driver2}_position_data.csv"
    data2.to_csv(filename2)
    print(f"Exported {driver2} data to: {filename2}")
    
    # Export combined data
    combined_data = pd.concat([
        data1.assign(Driver=driver1),
        data2.assign(Driver=driver2)
    ], ignore_index=True)
    
    combined_filename = f"{export_dir}/combined_position_data.csv"
    combined_data.to_csv(combined_filename, index=False)
    print(f"Exported combined data to: {combined_filename}")
    
    # Export metadata
    metadata = {
        "race": race,
        "year": year,
        "drivers": [driver1, driver2],
        "export_timestamp": datetime.now().isoformat(),
        "data_points": {
            driver1: len(data1),
            driver2: len(data2)
        }
    }
    
    metadata_filename = f"{export_dir}/metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported metadata to: {metadata_filename}")


def get_fastest_lap_data(session, driver_code: str) -> Optional[pd.DataFrame]:
    """
    Get position data for a driver's fastest qualifying lap.
    
    Args:
        session: FastF1 session object
        driver_code: Three-letter driver code
    
    Returns:
        DataFrame with fastest lap position data or None if not found
    """
    try:
        # Get the session results to find the driver
        results = session.results
        driver_result = results.loc[results['Abbreviation'] == driver_code]
        
        if driver_result.empty:
            print(f"Driver {driver_code} not found in session")
            return None
        
        # Get the driver number
        driver_number = driver_result.iloc[0]['DriverNumber']
        
        # Get all laps for this driver
        driver_laps = session.laps.pick_drivers(driver_number)
        
        if driver_laps.empty:
            print(f"No laps found for driver {driver_code}")
            return None
        
        # Find the fastest lap
        fastest_lap = driver_laps.pick_fastest()
        fastest_lap_time = fastest_lap['LapTime']
        
        print(f"Fastest lap for {driver_code}: {fastest_lap_time}")
        
        # Prefer telemetry which often contains X/Y; fall back to merging with position data
        telemetry = fastest_lap.get_telemetry()
        if telemetry is None or telemetry.empty:
            print(f"No telemetry found for fastest lap of {driver_code}")
            return None

        # Debug available columns
        print(f"Available columns in telemetry: {list(telemetry.columns)}")

        has_xy = ('X' in telemetry.columns) and ('Y' in telemetry.columns)
        if has_xy:
            fastest_lap_positions = pd.DataFrame({
                'X': telemetry['X'],
                'Y': telemetry['Y'],
                'Status': 'OnTrack'
            })
        else:
            # Try to merge with session position data using SessionTime or Date
            pos_df = session.pos_data.get(driver_number)
            if pos_df is None or pos_df.empty:
                print(f"Position data not available to reconstruct X/Y for {driver_code}")
                return None

            left_time_key = None
            right_time_key = None

            if 'SessionTime' in telemetry.columns:
                left_time_key = 'SessionTime'
            elif 'Date' in telemetry.columns:
                left_time_key = 'Date'

            if 'SessionTime' in pos_df.columns:
                right_time_key = 'SessionTime'
            elif 'Date' in pos_df.columns:
                right_time_key = 'Date'

            if not left_time_key or not right_time_key:
                print(f"Could not align telemetry and position data for {driver_code}")
                return None

            telem = telemetry[[left_time_key]].copy()
            pos = pos_df[[right_time_key, 'X', 'Y']].copy()

            telem_sorted = telem.sort_values(by=left_time_key)
            pos_sorted = pos.sort_values(by=right_time_key)

            merged = pd.merge_asof(
                telem_sorted,
                pos_sorted,
                left_on=left_time_key,
                right_on=right_time_key,
                direction='nearest'
            )

            if merged[['X', 'Y']].isna().all().all():
                print(f"Failed to merge telemetry with position data for {driver_code}")
                return None

            fastest_lap_positions = pd.DataFrame({
                'X': merged['X'],
                'Y': merged['Y'],
                'Status': 'OnTrack'
            })
        
        # Add driver information
        fastest_lap_positions['Driver'] = driver_code
        fastest_lap_positions['DriverNumber'] = driver_number
        fastest_lap_positions['LapTime'] = fastest_lap_time
        
        if fastest_lap_positions.empty:
            print(f"No position data found for fastest lap of {driver_code}")
            return None
        
        # Add driver information
        fastest_lap_positions = fastest_lap_positions.copy()
        fastest_lap_positions['Driver'] = driver_code
        fastest_lap_positions['DriverNumber'] = driver_number
        fastest_lap_positions['LapTime'] = fastest_lap_time
        
        print(f"Retrieved {len(fastest_lap_positions)} position data points for {driver_code}'s fastest lap")
        
        return fastest_lap_positions
        
    except Exception as e:
        print(f"Error getting fastest lap data for driver {driver_code}: {e}")
        return None


def create_track_plot(data1: pd.DataFrame, data2: pd.DataFrame, 
                     driver1: str, driver2: str, 
                     year: int, race: str, session=None, road: bool = False) -> None:
    """
    Create a static plot showing both drivers' fastest lap paths overlaid on the track.
    
    Args:
        data1: Position data for first driver
        data2: Position data for second driver
        driver1: First driver code
        driver2: Second driver code
        year: Year of the race
        race: Name of the race
        session: FastF1 session object for track layout
    """
    plt.figure(figsize=(15, 10))
    
    # Plot track layout if available
    if session is not None:
        try:
            track = session.track
            plt.plot(track.x, track.y, 'k-', linewidth=3, alpha=0.7, label='Track')
        except:
            pass
    
    # Filter data to only show on-track positions
    data1_track = data1[data1['Status'] == 'OnTrack']
    data2_track = data2[data2['Status'] == 'OnTrack']
    
    # Set colors based on drivers
    if driver1 == 'NOR':
        color1 = '#FFA500'  # Orange for NOR
    else:
        color1 = '#FFA500'  # Default orange
        
    if driver2 == 'PIA':
        color2 = '#505050'  # Dark grey for PIA
    else:
        color2 = '#505050'  # Default dark grey
    
    # Add road surface if requested
    if road:
        # Try to load track data first (for Hungary/Budapest and Spa)
        track_data = None
        if race.lower() in ['hungary', 'hungarian', 'budapest']:
            track_data = load_track_data('Budapest')
        elif race.lower() in ['spa', 'spa-francorchamps', 'belgium', 'belgian']:
            track_data = load_track_data('Spa')
        
        if track_data is not None:
            # Use actual track data to create road
            track_name = "Budapest" if race.lower() in ['hungary', 'hungarian', 'budapest'] else "Spa"
            print(f"Using {track_name} track data for road surface")
            create_road_from_track_data(track_data, data1_track, data2_track, plt)
        else:
            # Fallback: Create thick grey road based on both driver's racing lines
            road_width = 20  # Thick road surface
            
            # Combine both driver paths to create a road surface
            import numpy as np
            
            # Get all X,Y coordinates from both drivers
            all_x = np.concatenate([data1_track['X'].values, data2_track['X'].values])
            all_y = np.concatenate([data1_track['Y'].values, data2_track['Y'].values])
            
            # Create a convex hull or smoothed path for the road
            from scipy.spatial import ConvexHull
            
            # Remove duplicate points and sort by distance along path
            points = np.column_stack([all_x, all_y])
            unique_points = np.unique(points, axis=0)
            
            if len(unique_points) > 3:
                try:
                    # Create convex hull for road outline
                    hull = ConvexHull(unique_points)
                    hull_points = unique_points[hull.vertices]
                    
                    # Close the hull
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    
                    # Plot the road as a filled polygon
                    plt.fill(hull_points[:, 0], hull_points[:, 1], 
                            color='#808080', alpha=0.6, zorder=1, label='Road Surface')
                    
                    # Add road outline
                    plt.plot(hull_points[:, 0], hull_points[:, 1], 
                            color='#606060', linewidth=road_width/10, alpha=0.8, zorder=2)
                            
                except Exception as e:
                    print(f"Could not create road surface: {e}")
                    # Fallback: create simple road from driver paths
                    plt.plot(data1_track['X'], data1_track['Y'], 
                            color='#808080', linewidth=road_width, alpha=0.6, zorder=1, label='Road Surface')
                    plt.plot(data2_track['X'], data2_track['Y'], 
                            color='#808080', linewidth=road_width, alpha=0.6, zorder=1)
    
    # Plot driver paths (without markers, we'll add car images)
    plt.plot(data1_track['X'], data1_track['Y'], 
             color=color1, linewidth=3, alpha=0.9, 
             label=f'{driver1} Fastest Lap')
    
    plt.plot(data2_track['X'], data2_track['Y'], 
             color=color2, linewidth=3, alpha=0.9, 
             label=f'{driver2} Fastest Lap')
    
    # Add car images at start and end points
    ax = plt.gca()
    
    if len(data1_track) > 0:
        # Calculate direction for start and end points
        if len(data1_track) > 1:
            # Direction at start (from first to second point)
            start_direction = calculate_direction(
                data1_track['X'].iloc[0], data1_track['Y'].iloc[0],
                data1_track['X'].iloc[1], data1_track['Y'].iloc[1]
            )
            # Direction at end (from second-to-last to last point)
            end_direction = calculate_direction(
                data1_track['X'].iloc[-2], data1_track['Y'].iloc[-2],
                data1_track['X'].iloc[-1], data1_track['Y'].iloc[-1]
            )
        else:
            start_direction = end_direction = 0
        
        # Load car images with rotation (smaller, higher quality)
        car1_start_img = load_car_image(driver1, size=20, rotation=start_direction)
        car1_end_img = load_car_image(driver1, size=20, rotation=end_direction)
        
        if car1_start_img:
            # Add car at start
            ab1_start = AnnotationBbox(car1_start_img, (data1_track['X'].iloc[0], data1_track['Y'].iloc[0]), 
                                     frameon=False, zorder=10)
            ax.add_artist(ab1_start)
        
        if car1_end_img:
            # Add car at end
            ab1_end = AnnotationBbox(car1_end_img, (data1_track['X'].iloc[-1], data1_track['Y'].iloc[-1]), 
                                   frameon=False, zorder=10)
            ax.add_artist(ab1_end)
        else:
            # Fallback to markers if no car image
            plt.scatter(data1_track['X'].iloc[0], data1_track['Y'].iloc[0], 
                       color=color1, s=150, marker='o', edgecolor='black', linewidth=2,
                       label=f'{driver1} Lap Start', zorder=5)
            plt.scatter(data1_track['X'].iloc[-1], data1_track['Y'].iloc[-1], 
                       color=color1, s=150, marker='X', edgecolor='black', linewidth=2,
                       label=f'{driver1} Lap End', zorder=5)
    
    if len(data2_track) > 0:
        # Calculate direction for start and end points
        if len(data2_track) > 1:
            # Direction at start (from first to second point)
            start_direction = calculate_direction(
                data2_track['X'].iloc[0], data2_track['Y'].iloc[0],
                data2_track['X'].iloc[1], data2_track['Y'].iloc[1]
            )
            # Direction at end (from second-to-last to last point)
            end_direction = calculate_direction(
                data2_track['X'].iloc[-2], data2_track['Y'].iloc[-2],
                data2_track['X'].iloc[-1], data2_track['Y'].iloc[-1]
            )
        else:
            start_direction = end_direction = 0
        
        # Load car images with rotation (smaller, higher quality)
        car2_start_img = load_car_image(driver2, size=20, rotation=start_direction)
        car2_end_img = load_car_image(driver2, size=20, rotation=end_direction)
        
        if car2_start_img:
            # Add car at start
            ab2_start = AnnotationBbox(car2_start_img, (data2_track['X'].iloc[0], data2_track['Y'].iloc[0]), 
                                     frameon=False, zorder=10)
            ax.add_artist(ab2_start)
        
        if car2_end_img:
            # Add car at end
            ab2_end = AnnotationBbox(car2_end_img, (data2_track['X'].iloc[-1], data2_track['Y'].iloc[-1]), 
                                   frameon=False, zorder=10)
            ax.add_artist(ab2_end)
        else:
            # Fallback to markers if no car image
            plt.scatter(data2_track['X'].iloc[0], data2_track['Y'].iloc[0], 
                       color=color2, s=150, marker='s', edgecolor='black', linewidth=2,
                       label=f'{driver2} Lap Start', zorder=5)
            plt.scatter(data2_track['X'].iloc[-1], data2_track['Y'].iloc[-1], 
                       color=color2, s=150, marker='X', edgecolor='black', linewidth=2,
                       label=f'{driver2} Lap End', zorder=5)
    
    # Remove all UI elements for clean look
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.legend().set_visible(False)
    plt.grid(False)
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Save the plot
    filename = f"{year}_{race.replace(' ', '_')}_{driver1}_vs_{driver2}_fastest_laps.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved fastest lap plot: {filename}")
    
    plt.show()


def create_animated_track_plot(data1: pd.DataFrame, data2: pd.DataFrame, 
                              driver1: str, driver2: str, 
                              year: int, race: str, session=None,
                              follow: bool = False, window_size: float = 300.0,
                              gif_seconds: Optional[float] = None,
                              default_fps: int = 30, no_lines: bool = False, 
                              road: bool = False) -> None:
    """
    Create an animated plot showing both drivers' fastest lap positions over time.
    
    Args:
        data1: Position data for first driver
        data2: Position data for second driver
        driver1: First driver code
        driver2: Second driver code
        year: Year of the race
        race: Name of the race
        session: FastF1 session object for track layout
        follow: Whether to follow the cars with the camera
        window_size: Size of the follow window in meters
        gif_seconds: Target duration for the GIF in seconds
        default_fps: Default frames per second for animation
        no_lines: If True, make racing lines white instead of colored
        road: If True, add thick grey road surface based on driver racing lines
    """
    # Filter data to only show on-track positions
    data1_track = data1[data1['Status'] == 'OnTrack'].reset_index()
    data2_track = data2[data2['Status'] == 'OnTrack'].reset_index()
    
    if len(data1_track) == 0 or len(data2_track) == 0:
        print("No on-track data available for animation")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot track layout if available
    if session is not None:
        try:
            track = session.track
            ax.plot(track.x, track.y, 'k-', linewidth=3, alpha=0.7, label='Track')
        except:
            pass
    
    # Set colors based on drivers
    if no_lines:
        # Make lines white to hide them but keep cars visible
        color1 = 'white'
        color2 = 'white'
    else:
        if driver1 == 'NOR':
            color1 = '#FFA500'  # Orange for NOR
        else:
            color1 = '#FFA500'  # Default orange
            
        if driver2 == 'PIA':
            color2 = '#505050'  # Dark grey for PIA
        else:
            color2 = '#505050'  # Default dark grey
    
    # Set markers (always use colored markers for cars)
    marker1 = 'o'
    marker2 = 's'
    
    # Initialize car annotations (will be updated in animation)
    car1_annotation = None
    car2_annotation = None
    
    # Check if car images exist
    car1_exists = os.path.exists(f"assets/cars/{driver1.lower()}.png")
    car2_exists = os.path.exists(f"assets/cars/{driver2.lower()}.png")
    
    # Calculate car size based on follow mode (will be updated after road creation if needed)
    if follow:
        car_size = max(30, int(window_size / 3))
    else:
        car_size = 25
    
    # Calculate line width based on car size (1/3 car width as requested)
    # Use a fixed line width that doesn't change between follow and non-follow modes
    line_width = max(1, 80 / 3)  # Fixed line width based on standard car size
    
    # Add road surface if requested
    if road:
        # Try to load track data first (for Hungary/Budapest and Spa)
        track_data = None
        if race.lower() in ['hungary', 'hungarian', 'budapest']:
            track_data = load_track_data('Budapest')
        elif race.lower() in ['spa', 'spa-francorchamps', 'belgium', 'belgian']:
            track_data = load_track_data('Spa')
        
        if track_data is not None:
            # Use actual track data to create road
            track_name = "Budapest" if race.lower() in ['hungary', 'hungarian', 'budapest'] else "Spa"
            print(f"Using {track_name} track data for road surface")
            create_road_from_track_data(track_data, data1_track, data2_track, ax)
            
            # Calculate proper car size based on F1 car width (2m) and track scaling
            # Get average track width from the data
            avg_track_width = (track_data['w_tr_right_m'].mean() + track_data['w_tr_left_m'].mean())
            # F1 car is 2m wide, so calculate pixel size based on track width
            # Use a reasonable pixel density where track width maps to a good visual size
            target_track_width_pixels = 200  # Target track width in pixels for good visualization
            pixel_density = target_track_width_pixels / avg_track_width
            f1_car_width_pixels = 2.0 * pixel_density * 2  # 2m car width in pixels, doubled for visibility
            
            if follow:
                # Scale up for follow mode
                car_size = max(int(f1_car_width_pixels), int(window_size / 20))
            else:
                car_size = int(f1_car_width_pixels)
            
            print(f"Track width: {avg_track_width:.1f}m, Car size: {car_size}px")
            
            # Keep line width consistent (already set above)
            # line_width remains the same regardless of car size changes
        else:
            # Fallback: Create thick grey road based on both driver's racing lines
            road_width = max(20, car_size * 2)  # Much thicker than racing lines
            
            # Combine both driver paths to create a road surface
            import numpy as np
            
            # Get all X,Y coordinates from both drivers
            all_x = np.concatenate([data1_track['X'].values, data2_track['X'].values])
            all_y = np.concatenate([data1_track['Y'].values, data2_track['Y'].values])
            
            # Create a convex hull or smoothed path for the road
            from scipy.spatial import ConvexHull
            from scipy.interpolate import interp1d
            
            # Remove duplicate points and sort by distance along path
            points = np.column_stack([all_x, all_y])
            unique_points = np.unique(points, axis=0)
            
            if len(unique_points) > 3:
                try:
                    # Create convex hull for road outline
                    hull = ConvexHull(unique_points)
                    hull_points = unique_points[hull.vertices]
                    
                    # Close the hull
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    
                    # Plot the road as a filled polygon
                    ax.fill(hull_points[:, 0], hull_points[:, 1], 
                           color='#808080', alpha=0.6, zorder=1, label='Road Surface')
                    
                    # Add road outline
                    ax.plot(hull_points[:, 0], hull_points[:, 1], 
                           color='#606060', linewidth=road_width/10, alpha=0.8, zorder=2)
                           
                except Exception as e:
                    print(f"Could not create road surface: {e}")
                    # Fallback: create simple road from driver paths
                    ax.plot(data1_track['X'], data1_track['Y'], 
                           color='#808080', linewidth=road_width, alpha=0.6, zorder=1, label='Road Surface')
                    ax.plot(data2_track['X'], data2_track['Y'], 
                           color='#808080', linewidth=road_width, alpha=0.6, zorder=1)
    
    # Initialize empty lines
    line1, = ax.plot([], [], color=color1, linewidth=line_width, alpha=0.8, label=f'{driver1} Fastest Lap')
    line2, = ax.plot([], [], color=color2, linewidth=line_width, alpha=0.8, label=f'{driver2} Fastest Lap')
    
    # Create car image elements that will be updated in animation
    car1_image = None
    car2_image = None
    
    if car1_exists:
        # Create initial car image
        car1_img = load_car_image(driver1, size=car_size, rotation=0, maintain_aspect=follow)
        if car1_img:
            # Convert PIL image to numpy array for imshow
            import numpy as np
            car1_array = np.array(car1_img)
            # Get actual dimensions for proper extent
            img_height, img_width = car1_array.shape[:2]
            # Create a small image plot that we'll update
            car1_image = ax.imshow(car1_array, extent=[0, img_width, 0, img_height], 
                                 zorder=10, visible=False)
    
    if car2_exists:
        # Create initial car image
        car2_img = load_car_image(driver2, size=car_size, rotation=0, maintain_aspect=follow)
        if car2_img:
            # Convert PIL image to numpy array for imshow
            import numpy as np
            car2_array = np.array(car2_img)
            # Get actual dimensions for proper extent
            img_height, img_width = car2_array.shape[:2]
            # Create a small image plot that we'll update
            car2_image = ax.imshow(car2_array, extent=[0, img_width, 0, img_height], 
                                 zorder=10, visible=False)
    
    # Fallback markers if no car images
    point1, = ax.plot([], [], color=color1, marker=marker1, markersize=12, label=f'{driver1} Current', visible=not car1_exists)
    point2, = ax.plot([], [], color=color2, marker=marker2, markersize=12, label=f'{driver2} Current', visible=not car2_exists)
    
    # Set initial axis limits
    all_x = pd.concat([data1_track['X'], data2_track['X']])
    all_y = pd.concat([data1_track['Y'], data2_track['Y']])

    if follow:
        # Center on initial positions
        x_points = []
        y_points = []
        if len(data1_track) > 0:
            x_points.append(float(data1_track['X'].iloc[0]))
            y_points.append(float(data1_track['Y'].iloc[0]))
        if len(data2_track) > 0:
            x_points.append(float(data2_track['X'].iloc[0]))
            y_points.append(float(data2_track['Y'].iloc[0]))
        if x_points and y_points:
            cx = sum(x_points) / len(x_points)
            cy = sum(y_points) / len(y_points)
            pad = window_size * 1.2
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)
        else:
            ax.set_xlim(all_x.min() - 50, all_x.max() + 50)
            ax.set_ylim(all_y.min() - 50, all_y.max() + 50)
    else:
        ax.set_xlim(all_x.min() - 50, all_x.max() + 50)
        ax.set_ylim(all_y.min() - 50, all_y.max() + 50)
    # Remove all UI elements for clean look
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.legend().set_visible(False)
    ax.grid(False)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Determine fps/interval to target desired gif duration if provided
    max_frames = max(len(data1_track), len(data2_track))
    fps = default_fps
    if gif_seconds and gif_seconds > 0:
        fps = max(1, int(round(max_frames / gif_seconds)))
    interval_ms = int(round(1000.0 / fps))

    # Animation function
    def animate(frame):
        # Update lines (show path up to current frame)
        if frame < len(data1_track):
            line1.set_data(data1_track['X'].iloc[:frame+1], data1_track['Y'].iloc[:frame+1])
            # Update car position or fallback marker
            if car1_image and car1_exists:
                # Calculate direction of movement
                if frame > 0:
                    direction = calculate_direction(
                        data1_track['X'].iloc[frame-1], data1_track['Y'].iloc[frame-1],
                        data1_track['X'].iloc[frame], data1_track['Y'].iloc[frame]
                    )
                    # Update car image with new rotation (use scaled size)
                    new_car_img = load_car_image(driver1, size=car_size, rotation=direction, maintain_aspect=follow)
                    if new_car_img:
                        import numpy as np
                        car1_array = np.array(new_car_img)
                        car1_image.set_array(car1_array)
                        
                        # Update extent based on actual image dimensions
                        img_height, img_width = car1_array.shape[:2]
                        x_pos = data1_track['X'].iloc[frame]
                        y_pos = data1_track['Y'].iloc[frame]
                        car1_image.set_extent([x_pos - img_width/2, x_pos + img_width/2, 
                                             y_pos - img_height/2, y_pos + img_height/2])
                else:
                    # Update position - center the car image at the current position
                    x_pos = data1_track['X'].iloc[frame]
                    y_pos = data1_track['Y'].iloc[frame]
                    car1_image.set_extent([x_pos - car_size/2, x_pos + car_size/2, 
                                         y_pos - car_size/2, y_pos + car_size/2])
                car1_image.set_visible(True)
            else:
                point1.set_data([data1_track['X'].iloc[frame]], [data1_track['Y'].iloc[frame]])
        
        if frame < len(data2_track):
            line2.set_data(data2_track['X'].iloc[:frame+1], data2_track['Y'].iloc[:frame+1])
            # Update car position or fallback marker
            if car2_image and car2_exists:
                # Calculate direction of movement
                if frame > 0:
                    direction = calculate_direction(
                        data2_track['X'].iloc[frame-1], data2_track['Y'].iloc[frame-1],
                        data2_track['X'].iloc[frame], data2_track['Y'].iloc[frame]
                    )
                    # Update car image with new rotation (use scaled size)
                    new_car_img = load_car_image(driver2, size=car_size, rotation=direction, maintain_aspect=follow)
                    if new_car_img:
                        import numpy as np
                        car2_array = np.array(new_car_img)
                        car2_image.set_array(car2_array)
                        
                        # Update extent based on actual image dimensions
                        img_height, img_width = car2_array.shape[:2]
                        x_pos = data2_track['X'].iloc[frame]
                        y_pos = data2_track['Y'].iloc[frame]
                        car2_image.set_extent([x_pos - img_width/2, x_pos + img_width/2, 
                                             y_pos - img_height/2, y_pos + img_height/2])
                else:
                    # Update position - center the car image at the current position
                    x_pos = data2_track['X'].iloc[frame]
                    y_pos = data2_track['Y'].iloc[frame]
                    car2_image.set_extent([x_pos - car_size/2, x_pos + car_size/2, 
                                         y_pos - car_size/2, y_pos + car_size/2])
                car2_image.set_visible(True)
            else:
                point2.set_data([data2_track['X'].iloc[frame]], [data2_track['Y'].iloc[frame]])

        if follow:
            # Recenter axis around current positions, ensuring both cars are always visible
            x_points = []
            y_points = []
            if frame < len(data1_track):
                x_points.append(float(data1_track['X'].iloc[frame]))
                y_points.append(float(data1_track['Y'].iloc[frame]))
            if frame < len(data2_track):
                x_points.append(float(data2_track['X'].iloc[frame]))
                y_points.append(float(data2_track['Y'].iloc[frame]))
            
            if x_points and y_points:
                # Calculate the bounding box that includes both cars
                min_x, max_x = min(x_points), max(x_points)
                min_y, max_y = min(y_points), max(y_points)
                
                # Calculate the center of the bounding box
                cx = (min_x + max_x) / 2
                cy = (min_y + max_y) / 2
                
                # Calculate the required window size to fit both cars
                required_width = max_x - min_x
                required_height = max_y - min_y
                
                # Use the larger of the required size or the specified window_size
                # Add padding to ensure cars aren't at the edge
                pad_x = max(window_size, required_width * 1.5) / 2
                pad_y = max(window_size, required_height * 1.5) / 2
                
                ax.set_xlim(cx - pad_x, cx + pad_x)
                ax.set_ylim(cy - pad_y, cy + pad_y)
        
        # Return all animated elements
        elements = [line1, line2, point1, point2]
        if car1_image:
            elements.append(car1_image)
        if car2_image:
            elements.append(car2_image)
        return elements
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                 interval=interval_ms, blit=(not follow), repeat=True)
    
    # Save animation
    filename = f"{year}_{race.replace(' ', '_')}_{driver1}_vs_{driver2}_fastest_laps_animated.gif"
    anim.save(filename, writer='pillow', fps=fps)
    print(f"Saved fastest lap animation: {filename}")
    
    plt.show()


def main():
    """Main function to handle command line arguments and execute the script."""
    parser = argparse.ArgumentParser(
        description="Extract F1 track position data for two drivers during qualifying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qualifying_positions.py --year 2024 --race "Monaco" --driver1 "VER" --driver2 "HAM"
  python qualifying_positions.py --year 2023 --race "Silverstone" --driver1 "LEC" --driver2 "NOR" --export
  python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --plot
  python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --animate
        """
    )
    
    parser.add_argument('--year', type=int, required=True,
                       help='Year of the race (e.g., 2024)')
    parser.add_argument('--race', type=str, required=True,
                       help='Name of the Grand Prix (e.g., "Monaco", "Silverstone")')
    parser.add_argument('--driver1', type=str, required=True,
                       help='Three-letter driver code for first driver (e.g., "VER", "HAM")')
    parser.add_argument('--driver2', type=str, required=True,
                       help='Three-letter driver code for second driver (e.g., "LEC", "NOR")')
    parser.add_argument('--export', action='store_true',
                       help='Export position data to CSV files')
    parser.add_argument('--plot', action='store_true',
                       help='Create static plot of fastest lap track positions')
    parser.add_argument('--animate', action='store_true',
                       help='Create animated plot of fastest lap track positions')
    parser.add_argument('--follow', action='store_true',
                       help='Enable camera follow: keep axes centered on cars')
    parser.add_argument('--follow-window', type=float, default=200.0,
                       help='Half-size of follow window in meters (default: 200)')
    parser.add_argument('--gif-seconds', type=float,
                        help='Target GIF duration in seconds (auto-adjust fps)')
    parser.add_argument('--no-lines', action='store_true',
                        help='Make racing lines white instead of colored (hide lines but keep cars)')
    parser.add_argument('--road', action='store_true',
                        help='Add thick grey road surface based on driver racing lines')
    
    args = parser.parse_args()
    
    # Validate year
    if args.year < 2018 or args.year > 2024:
        print(f"Warning: Year {args.year} may not have complete data available.")
        print("FastF1 typically has data from 2018 onwards.")
    
    # Convert driver codes to uppercase
    driver1 = args.driver1.upper()
    driver2 = args.driver2.upper()
    
    print(f"Extracting track position data for {driver1} vs {driver2}")
    print(f"Race: {args.year} {args.race}")
    
    # Get track position data
    data1, data2 = get_track_position_data(
        args.year, args.race, driver1, driver2
    )
    
    # Display results
    if data1 is not None and data2 is not None:
        analyze_position_data(driver1, data1, driver2, data2)
        
        if args.export:
            export_position_data(driver1, data1, driver2, data2, args.year, args.race)
        
        if args.plot or args.animate:
            # Load session for track layout and fastest lap data
            try:
                cache_dir = 'cache'
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                fastf1.Cache.enable_cache(cache_dir)
                
                session = fastf1.get_session(args.year, args.race, 'Q')
                session.load()
                
                # Get fastest lap data instead of all position data
                print(f"\nExtracting fastest lap data...")
                fastest_lap_data1 = get_fastest_lap_data(session, driver1)
                fastest_lap_data2 = get_fastest_lap_data(session, driver2)
                
                if fastest_lap_data1 is not None and fastest_lap_data2 is not None:
                    if args.animate:
                        create_animated_track_plot(
                            fastest_lap_data1, fastest_lap_data2,
                            driver1, driver2, args.year, args.race, session,
                            follow=args.follow, window_size=args.follow_window,
                            gif_seconds=args.gif_seconds, no_lines=args.no_lines,
                            road=args.road
                        )
                    else:
                        create_track_plot(fastest_lap_data1, fastest_lap_data2, driver1, driver2, args.year, args.race, session, road=args.road)
                else:
                    print("Could not extract fastest lap data for plotting")
                    
            except Exception as e:
                print(f"Error loading session for plotting: {e}")
                session = None
    else:
        print("\n❌ Failed to retrieve position data.")
        print("Please check:")
        print("- Year and race name are correct")
        print("- Driver codes are valid (3-letter format)")
        print("- Internet connection is available")
        sys.exit(1)


if __name__ == "__main__":
    main()
